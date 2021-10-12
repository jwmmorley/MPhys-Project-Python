"""Entry points for the console scripts."""

import os
import sys
from threading import Timer
import webbrowser

from . import __version__, app
from .processing import find_files


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print('usage: regression [run] [config] [path] [files] [version]')

    for command in args:
        if command in ['run', '-r', '--run']:
            Timer(1, webbrowser.open_new('http://localhost:8080/')).start()
            app.run(host='0.0.0.0', port=8080, debug=False)
        if command in ['config', '-c', '--config']:
            config_file = os.path.join(os.path.dirname(__file__), 'config.cfg')
            webbrowser.open(config_file)
        if command in ['path', '-p', '--path']:
            print(os.path.join(os.path.dirname(__file__)))
        if command in ['files', '-f', '--files']:
            print(list(find_files()))
        if command in ['version', '-v', '--version']:
            print(__version__)


if __name__ == '__main__':
    main()
