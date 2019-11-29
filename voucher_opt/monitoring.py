import hfds
from hfds.config import INFLUXDB_DATABASE, INFLUXDB_HOST

from voucher_opt.logger import log


# The stats in InfluxDB are consumed by dashboards in Grafana

class InfluxExporter:
    def __init__(self, country, elaboration_date, model_id):
        self.country = country
        self.elaboration_date = elaboration_date
        self.model_id = model_id

    def send_action_distribution(self, model_action_distribution):
        log.info(f'Exporting action distribution to InfluxDB, {INFLUXDB_HOST}:{INFLUXDB_DATABASE}')
        points = []
        for key, value in model_action_distribution.to_dict().items():
            points.append({
                'measurement': 'happy_action',
                'tags': {
                    'country': self.country,
                    'model_id': self.model_id,
                    'action': key
                },
                'fields': {
                    'value': value
                },
                'time': self.elaboration_date.isoformat()
            })
        hfds.grafana.send_to_influxdb(points)

    def send_event_stats(self, events_per_date):
        log.info(f'Exporting event statistics to InfluxDB, {INFLUXDB_HOST}:{INFLUXDB_DATABASE}')
        points = []
        for key, value in events_per_date.to_dict().items():
            points.append({
                'measurement': 'happy_events',
                'tags': {
                    'country': self.country,
                    'model_id': self.model_id
                },
                'fields': {
                    'count': value
                },
                'time': key.isoformat()
            })
        hfds.grafana.send_to_influxdb(points)


def send_all_event_stats(events_per_country_date):
    log.info(f'Exporting event statistics to InfluxDB, {INFLUXDB_HOST}:{INFLUXDB_DATABASE}')
    points = []
    for key, value in events_per_country_date.to_dict().items():
        points.append({
            'measurement': 'happy_all_events',
            'tags': {
                'country': key[0]
            },
            'fields': {
                'count': value
            },
            'time': key[1].isoformat()
        })
    hfds.grafana.send_to_influxdb(points)
