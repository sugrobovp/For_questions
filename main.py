# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals, print_function, division

import statistics
from typing import Union, Optional, Any, Tuple

from django.db.models import Exists

from vist_core.logic.soppage_logic import BaseStoppage, UseRawDataCache
from vist_domain.query import TruckActiveIdsQuery, TypedStaticObjectIdsQuery
from vist_framework.utils import old_div

import time
import numbers
from datetime import datetime, timedelta
from collections import defaultdict

from django import forms
from django.utils.translation import ugettext_lazy as _

from vist_domain.entity import VehicleId

from vist_core.logic.fuel_transition import FuelTransitionLogic
from vist_core.logic.procedure_settings import ProcedureSettings, ProcedureSettingsGroup
from vist_core.management.base_procedure import BaseCommand, BaseMessageProcedure, ProcedureSettingsMixin
from vist_core.models import CheckpointTransition
from vist_core.utils import parse_utc, InstanceBulker

from vist_pit.models import Trip
from vist_pit.models.road_graph import CheckpointStats, CheckpointStatsArchive
from vist_pit.logic.reduced_length import get_reduced_length_params

INITIAL_SLOW_MOVE = 10
INITIAL_MIN_DURATION = 0
INITIAL_MAX_DURATION = 3600
INITIAL_SAMPLE_COUNT = 20
INITIAL_SAMPLE_PERIOD = 0


class SlowMoveSettingsForm(forms.Form):
    load_slow_speed_kmph = forms.IntegerField(
        label=_('Скорость движения под погрузку, считаемая медленной, км/ч'),
        initial=INITIAL_SLOW_MOVE,
        min_value=1,
    )
    unload_slow_speed_kmph = forms.IntegerField(
        label=_('Скорость движения перед разгрузкой, считаемая медленной, км/ч'),
        initial=INITIAL_SLOW_MOVE,
        min_value=1,
    )
    static_slow_speed_kmph = forms.IntegerField(
        label=_('Скорость движения в статических зонах, считаемая медленной, км/ч'),
        initial=INITIAL_SLOW_MOVE,
        min_value=1,
    )
    checkpoint_slow_speed_kmph = forms.IntegerField(
        label=_('Скорость движения между чекпоинтами, считаемая медленной, км/ч'),
        initial=INITIAL_SLOW_MOVE,
        min_value=1,
    )


class MedianSettingsForm(forms.Form):
    min_duration = forms.IntegerField(
        label=_('Минимальная длительность перемещения, с'),
        initial=INITIAL_MIN_DURATION,
        min_value=0,
    )
    max_duration = forms.IntegerField(
        label=_('Максимальная длительность перемещения, с'),
        initial=INITIAL_MAX_DURATION,
        min_value=1,
    )
    sample_count = forms.IntegerField(
        label=_('Количество записей в выборке'),
        initial=INITIAL_SAMPLE_COUNT,
        min_value=1,
    )
    sample_period = forms.IntegerField(
        label=_('Период выборки, дней (0 - не ограничен)'),
        initial=INITIAL_SAMPLE_PERIOD,
        min_value=0,
    )


class SlowMoveSettingsGroup(ProcedureSettingsGroup):
    name = 'slow_move'
    title = _('Алгоритм фильтрации медленного движение и движения задним ходом к Экскаваторам')
    form_class = SlowMoveSettingsForm
    template = '/static/pit/procedure-settings/ps-checkpoint-statistics-slow-move.html'


class MedianSettingsGroup(ProcedureSettingsGroup):
    name = 'median'
    title = _('Настройки медианного фильтра для расчета агрегрированных статистик')
    form_class = MedianSettingsForm
    template = '/static/pit/procedure-settings/ps-checkpoint-statistics-median.html'


class CheckpointStatisticsProcedureSettings(ProcedureSettings):
    name = 'checkpoint_statistics'
    title = _('Процедура статистики движения самосвалов по контрольным точкам')
    settings_groups = [
        SlowMoveSettingsGroup,
        MedianSettingsGroup,
    ]


class CheckpointStatistics(ProcedureSettingsMixin, BaseMessageProcedure):
    """
        Статистика движения самосвалов по контрольным точкам.
    """
    verbose_name = _('Статистика движения самосвалов по контрольным точкам')
    tags = ['checkpoints']

    source = [
        'CheckpointTransitions',
        'SetToLoadUnloadStatistic',
        'GeoWatch',  # geo-zone intersections
        ("Stoppage", BaseStoppage.EventType.STOPPAGE_BEGIN),
        ("Stoppage", BaseStoppage.EventType.STOPPAGE_END),
        ("RawStoppages", BaseStoppage.EventType.STOPPAGE_BEGIN),
        ("RawStoppages", BaseStoppage.EventType.STOPPAGE_END),
    ]
    extra_apps = "core,pit"

    AVG_WEIGHT = 3
    caches_lifetime = 300
    save_archive_timeout = 60
    settings_class = CheckpointStatisticsProcedureSettings

    def __init__(self):
        super(CheckpointStatistics, self).__init__()
        self.truck_ids = []
        self.archive_stats = []
        self.unload_cache = defaultdict(list)
        self.static_cache = defaultdict(list)
        self.cache_use_raw = UseRawDataCache.get_instance()

        self.timer = time.time()
        self.save_archive_timer = time.time()
        self.load_slow_speed_mps = float(self.settings_load_slow_speed_kmph) / 3.6
        self.unload_slow_speed_mps = float(self.settings_unload_slow_speed_kmph) / 3.6
        self.static_slow_speed_mps = float(self.settings_static_slow_speed_kmph) / 3.6
        self.checkpoint_slow_speed_mps = float(self.settings_checkpoint_slow_speed_kmph) / 3.6
        self.min_duration = self.settings_min_duration
        self.max_duration = self.settings_max_duration
        self.refresh_caches()

    @property
    def settings_load_slow_speed_kmph(self):
        try:
            return self.procedure_settings.slow_move.load_slow_speed_kmph
        except (AttributeError, ) as e:
            return INITIAL_SLOW_MOVE

    @property
    def settings_unload_slow_speed_kmph(self):
        try:
            return self.procedure_settings.slow_move.unload_slow_speed_kmph
        except (AttributeError, ) as e:
            return INITIAL_SLOW_MOVE

    @property
    def settings_static_slow_speed_kmph(self):
        try:
            return self.procedure_settings.slow_move.static_slow_speed_kmph
        except (AttributeError, ) as e:
            return INITIAL_SLOW_MOVE

    @property
    def settings_checkpoint_slow_speed_kmph(self):
        try:
            return self.procedure_settings.slow_move.checkpoint_slow_speed_kmph
        except (AttributeError, ) as e:
            return INITIAL_SLOW_MOVE

    @property
    def settings_min_duration(self):
        try:
            return self.procedure_settings.median.min_duration
        except (AttributeError, ) as e:
            return INITIAL_MIN_DURATION

    @property
    def settings_max_duration(self):
        try:
            return self.procedure_settings.median.max_duration
        except (AttributeError, ) as e:
            return INITIAL_MAX_DURATION

    @property
    def settings_sample_count(self):
        try:
            return self.procedure_settings.median.sample_count
        except (AttributeError, ) as e:
            return INITIAL_SAMPLE_COUNT

    @property
    def settings_sample_period(self):
        try:
            return self.procedure_settings.median.sample_period
        except (AttributeError, ) as e:
            return INITIAL_SAMPLE_PERIOD

    def refresh_caches(self):
        self.truck_ids = TruckActiveIdsQuery().fetch()
        self.timer = time.time()

    def avg(self, old_value, new_value):
        return old_div(((self.AVG_WEIGHT - 1) * old_value + new_value), self.AVG_WEIGHT)

    @staticmethod
    def is_static(zone_id):
        """
        функция для проверки "Статический ли этот геообъект?"
        """
        return zone_id in TypedStaticObjectIdsQuery(active=True).fetch()

    # noinspection PyMethodMayBeStatic
    def get_parameters(
        self,  # type: CheckpointStatistics
        vehicle_id,  # type: VehicleId
        time_from,  # type: datetime
        time_to,  # type: datetime
        slow_speed,  # type: Optional[Union[int, float]]
        height_from,  # type: Optional[Any]
        height_to  # type: Optional[Any]
    ):  # type: (...) -> Tuple[int, int, int, int, int]
        """
            Получить duration, distance и параметры для расчёта приведенного расстояния при движении со скорость не менее slow_speed
        """
        period_duration = (time_to - time_from).total_seconds()
        distance, duration = 0, 0  # type: int, int

        if period_duration <= INITIAL_MAX_DURATION:
            distance, duration = FuelTransitionLogic.get_distance_and_duration_by_archive(
                vehicle_id, time_from, time_to, slow_speed=slow_speed, expand_time=True
            )  # type: int, int

        # truck = Truck.objects.get(pk=vehicle_id)
        raises, slopes, rotate_count = get_reduced_length_params(vehicle_id, time_from, time_to)  # type: int, int, int

        return duration, distance, raises, slopes, rotate_count

    def create_stat_filter(self, stat):
        f = stat.the_same_filter()
        if self.settings_sample_period:
            f['message_time__gte'] = datetime.now() - timedelta(days=self.settings_sample_period)
            f['message_time__lte'] = datetime.now()
        return f

    @staticmethod
    def calculate_median(l):
        l_cleaned = [val for val in l if isinstance(val, numbers.Real)]
        return None if not l_cleaned else statistics.median(l_cleaned)

    def calculate_medians(self, stat, duration, distance, raises, slopes, rotate_count):
        if self.settings_sample_period:
            now_time = datetime.now()

            def predicate(itm):
                return now_time - timedelta(days=self.settings_sample_period) <= itm.message_time <= now_time
        else:
            def predicate(itm):
                return True

        # последние записи из кэша
        # TODO Ахтунг!!! self.archive_stats очищается каждую минуту.
        # TODO Сделать нормальный кэш, инициализируемый на глубину self.settings_sample_period
        stats = [item for item in self.archive_stats if predicate(item) and item.the_same(stat)]
        stats.sort(key=lambda x: x.message_time, reverse=True)
        stats = stats[:self.settings_sample_count - 1]

        # дополняем данными из базы, если не хватает для выборки
        if len(stats) < self.settings_sample_count - 1:
            stat_filter = self.create_stat_filter(stat)
            stats.extend(CheckpointStatsArchive.objects.select_related('truck')
                         .filter(**stat_filter).order_by('-message_time')[:self.settings_sample_count - len(stats)])

        # выбираем длительности с учётом последних данных и вычисляем медиану
        durations = [item.duration for item in stats]
        durations.append(duration)

        # выбираем дистанции с учётом последних данных и вычисляем медиану
        distances = [item.distance for item in stats]
        distances.append(distance)

        raises_items = [item.raises for item in stats]
        raises_items.append(raises)

        slopes_items = [item.slopes for item in stats]
        slopes_items.append(slopes)

        rotate_count_items = [item.rotate_count for item in stats]
        rotate_count_items.append(rotate_count)

        return CheckpointStatistics.calculate_median(durations), CheckpointStatistics.calculate_median(distances), CheckpointStatistics.calculate_median(raises_items), CheckpointStatistics.calculate_median(slopes_items), CheckpointStatistics.calculate_median(rotate_count_items), len(durations)

    def add_stat(self, **kwargs):
        self.debug('Add stat: %s', kwargs)
        created = False

        if not kwargs.get('duration'):
            return created

        duration = kwargs['duration']
        distance = kwargs['distance']
        raises = kwargs['raises']
        slopes = kwargs['slopes']
        rotate_count = kwargs['rotate_count']

        if duration is None or distance is None:
            return created

        if duration < self.min_duration or duration > self.max_duration:
            return created

        try:
            stat = created = None
            if 'vertex_in' in kwargs and 'vertex_out' in kwargs:
                stat, created = CheckpointStats.objects.select_related('truck').get_or_create(
                    truck_id=kwargs['truck'],
                    is_loaded=kwargs.get('is_loaded', False),
                    vertex_in_id=kwargs.get('vertex_in'),
                    vertex_out_id=kwargs.get('vertex_out'),
                    shov_id=kwargs.get('shov', None),
                    unload_id=kwargs.get('unload', None),
                    static_id=kwargs.get('static', None),
                    defaults=dict(
                        message_time=kwargs.get('message_time'),
                        avg_duration=duration,
                        avg_distance=distance,
                        avg_raises=raises,
                        avg_slopes=slopes,
                        avg_rotate_count=rotate_count,
                    )
                )

            elif 'shov' in kwargs and bool(kwargs.get('vertex_in')) is not bool(kwargs.get('vertex_out')):
                stat, created = CheckpointStats.objects.select_related('truck').get_or_create(
                    truck_id=kwargs['truck'],
                    is_loaded=kwargs.get('is_loaded', False),
                    vertex_in_id=kwargs.get('vertex_in'),
                    vertex_out_id=kwargs.get('vertex_out'),
                    shov_id=kwargs.get('shov'),
                    unload=None,
                    static=None,
                    defaults=dict(
                        message_time=kwargs.get('message_time'),
                        avg_duration=duration,
                        avg_distance=distance,
                        avg_raises=raises,
                        avg_slopes=slopes,
                        avg_rotate_count=rotate_count,
                    )
                )

            elif 'unload' in kwargs and bool(kwargs.get('vertex_in')) is not bool(kwargs.get('vertex_out')):
                stat, created = CheckpointStats.objects.select_related('truck').get_or_create(
                    truck_id=kwargs['truck'],
                    is_loaded=kwargs.get('is_loaded', False),
                    vertex_in_id=kwargs.get('vertex_in'),
                    vertex_out_id=kwargs.get('vertex_out'),
                    shov=None,
                    unload_id=kwargs.get('unload'),
                    static=None,
                    defaults=dict(
                        message_time=kwargs.get('message_time'),
                        avg_duration=duration,
                        avg_distance=distance,
                        avg_raises=raises,
                        avg_slopes=slopes,
                        avg_rotate_count=rotate_count,
                    )
                )

            elif 'static' in kwargs and bool(kwargs.get('vertex_in')) is not bool(kwargs.get('vertex_out')):
                stat, created = CheckpointStats.objects.select_related('truck').get_or_create(
                    truck_id=kwargs['truck'],
                    is_loaded=kwargs.get('is_loaded', False),
                    vertex_in_id=kwargs.get('vertex_in'),
                    vertex_out_id=kwargs.get('vertex_out'),
                    shov=None,
                    unload=None,
                    static_id=kwargs.get('static'),
                    defaults=dict(
                        message_time=kwargs.get('message_time'),
                        avg_duration=duration,
                        avg_distance=distance,
                        avg_raises=raises,
                        avg_slopes=slopes,
                        avg_rotate_count=rotate_count,
                    )
                )

            if stat:
                if not created:
                    stat.message_time = kwargs.get('message_time')
                    mdn_duration, mdn_distance, mdn_raises, mdn_slopes, mdn_rotate_count, sample_count = self.calculate_medians(stat, duration, distance, raises, slopes, rotate_count)
                    stat.avg_duration = mdn_duration
                    stat.avg_distance = mdn_distance
                    stat.avg_raises = mdn_raises
                    stat.avg_slopes = mdn_slopes
                    stat.avg_rotate_count = mdn_rotate_count
                    stat.sample_count = sample_count
                    stat.save()

                self.archive_stats.append(CheckpointStatsArchive(
                    truck_id=kwargs['truck'],
                    is_loaded=kwargs.get('is_loaded', False),
                    vertex_in_id=kwargs.get('vertex_in'),
                    vertex_out_id=kwargs.get('vertex_out'),
                    shov_id=kwargs.get('shov'),
                    unload_id=kwargs.get('unload'),
                    static_id=kwargs.get('static'),
                    duration=duration,
                    distance=distance,
                    raises=raises,
                    slopes=slopes,
                    rotate_count=rotate_count,
                    avg_duration=stat.avg_duration,
                    avg_distance=stat.avg_distance,
                    avg_raises=stat.avg_raises,
                    avg_slopes=stat.avg_slopes,
                    avg_rotate_count=stat.avg_rotate_count,
                    sample_count=stat.sample_count,
                    message_time=kwargs.get('message_time'),
                ))
        except KeyError as e:
            self.error('Can`t save CheckpointStats without "%s"', e.message)

        return created

    def process_transition(self, vehicle_id, message_time, message):
        self.debug('Process transition')
        try:
            time_from = parse_utc(message['time_from'])
            time_to = parse_utc(message['time_to'])

            transition = CheckpointTransition.transitions.annotate(
                unload_trip_exists=Exists(Trip.objects.filter(
                    truck_id=vehicle_id,
                    end_time__range=(time_from, time_to),
                ))
            ).get(pk=message.get('transition_id'))

            through_stat = bool(transition.unload_trip_exists)  # признак статистики-индикатора сквозного проезда через объект

            # Проверяем заезд на статический объект
            for static_cache_entry in self.static_cache[vehicle_id]:
                enter_time = static_cache_entry['enter_time']
                leave_time = static_cache_entry['leave_time']
                first_stop_time = static_cache_entry['first_stop_time']
                last_stop_time = static_cache_entry['last_stop_time']
                first_stop_height = static_cache_entry['first_stop_height']
                last_stop_height = static_cache_entry['last_stop_height']

                # Заезд на статический объект произошёл в текущем транзишне
                # Статистика от чекпоинта до стат объекта
                if first_stop_time is not None and time_from < enter_time <= time_to:
                    duration, distance, raises, slopes, rotate_count = self.get_parameters(
                        vehicle_id,
                        time_from,
                        first_stop_time,
                        self.static_slow_speed_mps,
                        transition.height_from,
                        first_stop_height
                    )
                    self.add_stat(
                        truck=vehicle_id,
                        vertex_out=message['vertex_from'],
                        static=static_cache_entry['area_id'],
                        is_loaded=False,
                        duration=duration,
                        distance=distance,
                        raises=raises,
                        slopes=slopes,
                        rotate_count=rotate_count,
                        message_time=message_time,
                    )
                    # Чекпоинт, с которого заехали в статический объект
                    static_cache_entry['vertex_from'] = message['vertex_from']
                    static_cache_entry['vertex_from_time'] = time_from

                # Выезд из статического объекта произошёл в текущем транзишне
                # Статистика от стат объекта до чекпоинта
                if leave_time is not None and last_stop_time is not None and time_from < leave_time <= time_to:
                    duration, distance, raises, slopes, rotate_count = self.get_parameters(
                        vehicle_id,
                        last_stop_time,
                        time_to,
                        self.checkpoint_slow_speed_mps,
                        last_stop_height,
                        transition.height_to
                    )
                    self.add_stat(
                        truck=vehicle_id,
                        vertex_in=message['vertex_to'],
                        static=static_cache_entry['area_id'],
                        is_loaded=False,
                        duration=duration,
                        distance=distance,
                        raises=raises,
                        slopes=slopes,
                        rotate_count=rotate_count,
                        message_time=message_time
                    )
                    # Статистика индикатор сквозного проезда
                    if 'vertex_from' in static_cache_entry:
                        through_stat = True
                        duration, distance, raises, slopes, rotate_count = self.get_parameters(
                            vehicle_id,
                            static_cache_entry['vertex_from_time'],
                            time_to,
                            self.checkpoint_slow_speed_mps,
                            transition.height_from,
                            transition.height_to
                        )
                        self.add_stat(
                            truck=vehicle_id,
                            vertex_in=static_cache_entry['vertex_from'],
                            vertex_out=message['vertex_from'],
                            static=static_cache_entry['area_id'],
                            is_loaded=False,
                            duration=duration,
                            distance=distance,
                            raises=raises,
                            slopes=slopes,
                            rotate_count=rotate_count,
                            message_time=message_time
                        )

            # Удаляем устаревшие проезды по статическим объектам для самосвала
            self.static_cache[vehicle_id] = [x for x in self.static_cache[vehicle_id] if (x['enter_time'] < time_to - timedelta(days=1)) or (x['leave_time'] is not None and x['leave_time'] <= time_to)]

            # Статистика между чекпоинтами, если это не сквозной проезд через объект
            if not through_stat:
                duration, distance, raises, slopes, rotate_count = self.get_parameters(
                    vehicle_id,
                    time_from,
                    time_to,
                    self.checkpoint_slow_speed_mps,
                    transition.height_from,
                    transition.height_to
                )
                self.add_stat(
                    truck=vehicle_id,
                    vertex_in=message['vertex_to'],
                    vertex_out=message['vertex_from'],
                    is_loaded=int(message['weight']) > 10 if message['weight'] is not None else False,
                    duration=duration,
                    distance=distance,
                    raises=raises,
                    slopes=slopes,
                    rotate_count=rotate_count,
                    message_time=message_time,
                )

        except KeyError as e:
            self.error('Can`t find "%s" in message', e.message)

    def process_load(self, vehicle_id, message_time, message):
        self.debug('Process load')
        shov_id = message.get('shov')
        shov_height = message.get('height_load')
        if not shov_id:
            self.debug('Shov is undefined')
            return

        load_begin_time = message.get(
            'set_to_load_time_begin') or message.get(
            'load_arrive_time') or message.get(
            'begin_time')
        load_end_time = message.get('load_depart_time') or message.get('last_bucket_time') or message.get('begin_time')
        load_begin_time = parse_utc(load_begin_time) if load_begin_time else None
        load_end_time = parse_utc(load_end_time) if load_end_time else None
        if not load_begin_time and not load_end_time:
            self.debug('Loading time is undefined')
            return

        # Запрашиваем транзишн в котором была погрузка
        transition = CheckpointTransition.transitions.filter(
            vehicle_id=vehicle_id,
            time_from__lte=load_begin_time or load_end_time,
            time_to__gte=load_begin_time or load_end_time,
        ).order_by('time_from').first()

        if transition:
            # Статистика - индикатор сквозного проезда
            duration, distance, raises, slopes, rotate_count = self.get_parameters(
                vehicle_id,
                transition.time_from,
                transition.time_to,
                self.checkpoint_slow_speed_mps,
                transition.height_from,
                transition.height_to
            )
            self.add_stat(
                truck=vehicle_id,
                vertex_in=transition.vertex_to_id,
                vertex_out=transition.vertex_from_id,
                shov=shov_id,
                is_loaded=False,
                duration=duration,
                distance=distance,
                raises=raises,
                slopes=slopes,
                rotate_count=rotate_count,
                message_time=message_time,
            )

        if transition and load_begin_time:
            # Статистика от чекпоинта до экскаватора
            duration, distance, raises, slopes, rotate_count = self.get_parameters(
                vehicle_id,
                transition.time_from,
                load_begin_time,
                self.load_slow_speed_mps,
                transition.height_from,
                shov_height
            )
            self.add_stat(
                truck=vehicle_id,
                vertex_out=transition.vertex_from_id,
                shov=shov_id,
                is_loaded=False,
                duration=duration,
                distance=distance,
                raises=raises,
                slopes=slopes,
                rotate_count=rotate_count,
                message_time=message_time,
            )
        if transition and load_end_time:
            # Статистика от экскаватора до чекпоинта
            duration, distance, raises, slopes, rotate_count = self.get_parameters(
                vehicle_id,
                load_end_time,
                transition.time_to,
                self.checkpoint_slow_speed_mps,
                shov_height,
                transition.height_to
            )
            self.add_stat(
                truck=vehicle_id,
                vertex_in=transition.vertex_to_id,
                shov=shov_id,
                is_loaded=True,
                duration=duration,
                distance=distance,
                raises=raises,
                slopes=slopes,
                rotate_count=rotate_count,
                message_time=message_time,
            )

    def process_unload(self, vehicle_id, message_time, message):
        self.debug('Process unload')
        unload_id = message.get('unload')
        unload_height = message.get('height_unload')
        unload_time = message.get(
            'set_to_unload_time_begin') or message.get(
            'unload_arrive_time') or message.get(
            'end_time')  # type: datetime
        unload_time = parse_utc(unload_time) if unload_time else None
        if unload_id and unload_time:
            # Запрашиваем транзишн в котором была разгрузка
            transition = CheckpointTransition.transitions.filter(
                vehicle_id=vehicle_id,
                time_from__lte=unload_time,
                time_to__gte=unload_time,
            ).order_by('time_from').first()  # type: Optional[CheckpointTransition]

            if transition:
                if (unload_time - transition.time_from).total_seconds() > INITIAL_MAX_DURATION:
                    # если между проездом по чекпоинтам прошло очень много времени, очевидно,
                    # что нам вообще не нужна такая статистика
                    return
                # Статистика от чекпоинта до ПР
                duration_to_unload, distance_to_unload, raises_to_unload, slopes_to_unload, rotate_count_to_unload = self.get_parameters(
                    vehicle_id,
                    transition.time_from,
                    unload_time,
                    self.unload_slow_speed_mps,
                    transition.height_from,
                    unload_height
                )
                self.add_stat(
                    truck=vehicle_id,
                    vertex_out=transition.vertex_from_id,
                    unload=unload_id,
                    is_loaded=True,
                    duration=duration_to_unload,
                    distance=distance_to_unload,
                    raises=raises_to_unload,
                    slopes=slopes_to_unload,
                    rotate_count=rotate_count_to_unload,
                    message_time=message_time,
                )
                # Статистика от ПР до чекпоинта
                duration_from_unload, distance_from_unload, raises_from_unload, slopes_from_unload, rotate_count_from_unload = self.get_parameters(
                    vehicle_id,
                    unload_time,
                    transition.time_to,
                    self.checkpoint_slow_speed_mps,
                    unload_height,
                    transition.height_to
                )
                self.add_stat(
                    truck=vehicle_id,
                    vertex_in=transition.vertex_to_id,
                    unload=unload_id,
                    is_loaded=False,
                    duration=duration_from_unload,
                    distance=distance_from_unload,
                    raises=raises_from_unload,
                    slopes=slopes_from_unload,
                    rotate_count=rotate_count_from_unload,
                    message_time=message_time,
                )
                # Статистика сквозного проезда
                duration = duration_to_unload + duration_from_unload
                distance = distance_to_unload + distance_from_unload
                raises = raises_to_unload + raises_from_unload
                slopes = slopes_to_unload + slopes_from_unload
                rotate_count = rotate_count_from_unload
                self.add_stat(
                    truck=vehicle_id,
                    vertex_in=transition.vertex_to_id,
                    vertex_out=transition.vertex_from_id,
                    unload=unload_id,
                    is_loaded=False,
                    duration=duration,
                    distance=distance,
                    raises=raises,
                    slopes=slopes,
                    rotate_count=rotate_count,
                    message_time=message_time,
                )
            else:
                # Сюда не должны попадать, поскольку обеспечили синхронность с рейсами через SetToLoadUnloadStatistic
                self.warning('Can`t find CheckpointTransition for unloading')

    def process_geo(self, vehicle_id, message_time, message):
        self.debug('Process geo')
        time = message_time
        area_id = message['area_id']
        cross_type = message['cross_type']
        # работаем только со статическими зонами
        if not self.is_static(area_id):
            return
        self.debug('CS %s: get message %s', vehicle_id, message)
        veh_static_cache = self.static_cache[vehicle_id]
        # фиксируем зону статического объекта
        if cross_type:
            # вход в зону
            veh_event = dict(
                    area_id=area_id,
                    enter_time=time,
                    leave_time=None,
                    first_stop_time=None,
                    first_stop_height=None,
                    last_stop_time=None,
                    last_stop_height=None)
            veh_static_cache = [x for x in veh_static_cache if not(x['area_id'] == area_id and x['leave_time'] is None)]
            veh_static_cache.append(veh_event)
            self.static_cache[vehicle_id] = veh_static_cache
            self.debug('Process geo: veh_event start')
        else:
            # выход из зоны
            veh_events = [x for x in veh_static_cache if x['area_id'] == area_id and
                                x['enter_time'] < time and
                                x['leave_time'] is None]
            if len(veh_events) > 0:
                veh_events.sort(key=lambda x: x['enter_time'], reverse=True)
                veh_events[0]['leave_time'] = time
                self.debug('Process geo: veh_event end')

    def process_stoppage(self, vehicle_id, message_time, message):
        self.debug('Process stoppage')
        height = message.get('height')
        veh_static_cache = self.static_cache[vehicle_id]
        # начало простоя в зоне статического объекта
        veh_event = None
        # завершенные проезды по геозонам, в которые попадает простой
        veh_events = [x for x in veh_static_cache if x['leave_time'] and
                            x['enter_time'] <= message_time <= x['leave_time']]
        if len(veh_events) > 0:
            veh_events.sort(key=lambda x: x['enter_time'], reverse=True)
            veh_event = veh_events[0]
        else:
            # незавершенные проезды по геозонам, в которые попадает простой
            veh_events = [x for x in veh_static_cache if x['enter_time'] <= message_time and
                                x['leave_time'] is None]
            if len(veh_events) > 0:
                veh_events.sort(key=lambda x: x['enter_time'], reverse=True)
                veh_event = veh_events[0]
        if not veh_event:
            self.debug('Veh_event not found')
            return
        self.debug('CS %s: get message %s', vehicle_id, message)
        if message.get('event_type') == 'stoppage_begin':
            # первый простой в зоне статического объекта
            if veh_event['first_stop_time'] is None or veh_event['first_stop_time'] > message_time:
                veh_event['first_stop_time'] = message_time
                veh_event['first_stop_height'] = height
                self.debug('Process stoppage: veh_event first stop')
        # окончание простоя в зоне статического объекта
        elif message.get('event_type') == 'stoppage_end':
            # последний простой в зоне статического объекта
            if veh_event['last_stop_time'] is None or veh_event['last_stop_time'] < message_time:
                veh_event['last_stop_time'] = message_time
                veh_event['last_stop_height'] = height
                self.debug('Process stoppage: veh_event last stop')

    def _time(self, event_type, message):
        """Находит метку времени"""
        str_ = None
        if message['sender'] in ('RawStoppages', 'Stoppage'):
            str_ = message['time_begin'] if message['event_type'] == 'stoppage_begin' else message['time_end']
        elif message['sender'] == 'GeoWatch':
            str_ = message.get('cross_time')
        else:
            str_ = message.get('time', message.get('gmt_time', None))
        return parse_utc(str_) if str_ else None

    def get_message(self, message):
        vehicle_id = int(message["vehicle_id"])
        event_type = message.get('event_type', None)
        if vehicle_id not in self.truck_ids:
            return
        message_time = self._time(event_type, message)
        if message['sender'] == 'CheckpointTransitions':
            self.process_transition(vehicle_id, message_time, message)
        elif message['sender'] == 'SetToLoadUnloadStatistic':
            if message.get('set_to_load_checkpoint_id'):
                self.process_load(vehicle_id, message_time, message)
            if message.get('set_to_unload_checkpoint_id'):
                self.process_unload(vehicle_id, message_time, message)
        elif message['sender'] == 'GeoWatch':
            self.process_geo(vehicle_id, message_time, message)
        elif message['sender'] in ('RawStoppages', 'Stoppage'):
            if self.is_stop_message_applicable(vehicle_id, message['sender']):
                self.process_stoppage(vehicle_id, message_time, message)

    def is_stop_message_applicable(self, vehicle_id, sender):
        return any([
            self.cache_use_raw.is_raw(vehicle_id) and sender == "RawStoppages",
            not self.cache_use_raw.is_raw(vehicle_id) and sender == "Stoppage"
        ])

    def tic_tac(self, got_message=None, **kwargs):
        super(CheckpointStatistics, self).tic_tac(got_message=got_message, **kwargs)
        if time.time() - self.save_archive_timeout > self.save_archive_timer:
            self.save_archive()

    def on_get_message(self):
        if time.time() - self.caches_lifetime > self.timer:
            self.refresh_caches()

    def save_archive(self):
        recs_count = len(self.archive_stats)
        try:
            if recs_count > 0:
                self.debug('Save %s archive records', recs_count)
                with InstanceBulker(CheckpointStatsArchive, size=100) as csa:
                    for arh in self.archive_stats:
                        csa.push_instance(arh)

        except Exception as e:
            self.error('Exception while bulk_creating %s records of CheckpointStatsArchive: %s', recs_count, e)
        finally:
            self.archive_stats = []
            self.save_archive_timer = time.time()


class Command(BaseCommand):
    procedure_class = CheckpointStatistics