from datetime import datetime
from pathlib import Path
from wsgiref import simple_server

import falcon
import toml
from s3fs import S3FileSystem
from yattag import Doc

from config import internal_config
from voucher_opt.config.aws_parameters import AWSParameters
from voucher_opt.config.project_parameters import ProjectParameters
from voucher_opt.file_handling.cloud_utils import load_run_config_from_s3


# TODO: This is broken, fix when there is time

class ConfigResource:
    def __init__(self):
        print(f'Happy Hour config webapp running in {internal_config.ENVIRONMENT.name}')
        environment_config = internal_config.ENVIRONMENT_CONFIG[internal_config.ENVIRONMENT]
        self._aws_parameters = AWSParameters(environment_config)
        project_config = toml.load('project_config.toml')
        self._project_parameters = ProjectParameters(project_config)
        self._run_config = load_run_config_from_s3(self._project_parameters, self._aws_parameters)

    def write_run_config_to_s3(self, config_string):
        s3_key = self._project_parameters.compile_path({}, 'run_config', 'toml')
        s3_path = Path(s3_key)
        backup_path = Path(*s3_path.parts[:-1], f'run_config_until_{datetime.now()}.toml')
        with S3FileSystem().open(f'{self._aws_parameters.s3_config_bucket}/{backup_path}', 'wb') as f:
            f.write(toml.dumps(self._run_config).encode('utf-8'))
        with S3FileSystem().open(f'{self._aws_parameters.s3_config_bucket}/{s3_key}', 'wb') as f:
            f.write(config_string.encode('utf-8'))
        message = f'New config written to {self._aws_parameters.s3_config_bucket}/{s3_key}'
        print(message)
        return message

    def on_get(self, req, resp):
        self._run_config = load_run_config_from_s3(self._project_parameters, self._aws_parameters)

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'

        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag('head'):
                with tag('title'):
                    text('Happy Hour config')
            with tag('body', id='config'):
                with tag('h1'):
                    text('Happy Hour Config')
                with tag('form', action='/config', method='post'):
                    with tag('h2'):
                        text('Default config:')
                    doc.stag('br')
                    with tag('table', border=1):
                        for key, value in self._run_config['DEFAULT_CONFIG'].items():
                            with tag('tr'):
                                with tag('td', style='align:left'):
                                    text(f'{key}:\t')
                                with tag('td', style='align:left'):
                                    doc.input(name=f'default:{key}', type='text', value=f'{value}')
                    with tag('h2'):
                        text('Country specific config:')
                    for country_key, country_dict in self._run_config['COUNTRY_CONFIG'].items():
                        with tag('h3'):
                            text(f'Country {country_key}:')
                        with tag('table', border=1):
                            for key, value in country_dict.items():
                                with tag('tr'):
                                    with tag('td', style='align:left'):
                                        text(f'{key}:\t')
                                    with tag('td'):
                                        doc.input(name=f'country:{country_key}:{key}', type='text', value=f'{value}')
                    doc.stag('br')
                    doc.stag('br')
                    doc.stag('input',
                             type='submit',
                             value='Update config',
                             style='-webkit-appearance: button; height:50px; width:300px; font-size:20px')

        resp.body = (doc.getvalue())

    def on_post(self, req, resp):
        config = {}
        for key, value in req.params.items():
            parts = key.split(':')
            if parts[0] == 'default':
                try:
                    parsed_value = eval(value)
                except NameError:
                    parsed_value = value
                config.setdefault('DEFAULT_CONFIG', {})[parts[1]] = parsed_value
            elif parts[0] == 'country':
                try:
                    parsed_value = eval(value)
                except NameError:
                    parsed_value = value
                config.setdefault('COUNTRY_CONFIG', {}).setdefault(parts[1], {})[parts[2]] = parsed_value
        config_string = toml.dumps(config)
        print(config_string)
        message = self.write_run_config_to_s3(config_string)
        resp.status = falcon.HTTP_200
        resp.body = message


app = falcon.API()
app.req_options.auto_parse_form_urlencoded = True
app.add_route('/config', ConfigResource())

if __name__ == '__main__':
    httpd = simple_server.make_server('localhost', 5000, app)
    httpd.serve_forever()
