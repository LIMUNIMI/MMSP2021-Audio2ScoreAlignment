#! /usr/bin/env python3

from __future__ import print_function, unicode_literals

import json
import os
import pathlib
import tarfile
import tempfile
import time
from collections import deque
from ftplib import FTP
from getpass import getpass
from os.path import join as joinpath
from shutil import unpack_archive
from subprocess import DEVNULL, Popen
from urllib.parse import urlparse
from urllib.request import urlcleanup, urlretrieve

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.validation import Validator

from alive_progress import alive_bar
from mega import Mega
from pyfiglet import Figlet

from .asmd import load_definitions

#: Set to True to skip datasets which already exist
SKIP_EXISTING_DIR = True

LINK_GROUND_TRUTH = "https://mega.nz/file/3ct3gQoS#p-QXDNWVyHqB1puGt6Cr_1O8VZo9oF4qE7d-7yDqkfc"

THISDIR = os.path.dirname(os.path.realpath(__file__))

supported_archives = {
    '.zip': 'zip',
    '.tar': 'tar',
    '.tar.gz': 'gztar',
    '.tar.bz': 'bztar',
    '.tar.xz': 'xztar'
}


def chose_dataset(data, install_dir):
    """
    Ask for preferred datasets and removes unwanted from data. Also skips
    dataset with existing directories.
    """
    print("\nPlease, insert the number relative to the datasets that you want \
to download (ex. 1, 2, 3, 11, 2). If corresponding directories already exist, \
they will be skipped. Empty to select all of them.")
    for i, d in enumerate(data):
        print(i, "-", d['name'])

    flag = False
    while not flag:
        try:
            answer = prompt("\nWhich datasets do you want? ")
            # apply `int` function to each element in answer (characters) and
            # convert to list
            if answer == '':
                datalist = list(range(len(data)))
                flag = True
            else:
                datalist = list(map(int, answer.split(', ')))
                flag = True
        except ValueError:
            print("Wrong answer, please use only numbers in your answer.")
            flag = False

    # skipping directories already existing
    i = 0
    for k in range(len(data)):
        d = data[i]
        SKIP = os.path.isdir(os.path.join(install_dir,
                                          d['name'])) and SKIP_EXISTING_DIR
        if (SKIP) or (k not in datalist):
            print('Skipping ' + d['name'] +
                  ": already exists, or not selected!")
            del data[i]
            i -= 1
        i += 1


def definitions_path():
    validator = Validator.from_callable(
        lambda x: os.path.isdir(x) or x == '',
        error_message="Not a directory!",
        move_cursor_to_end=True,
    )
    path_completer = PathCompleter(only_directories=True)
    question = "\nType the path to a definition dir (empty for default and continue) "

    path = prompt(question, completer=path_completer, validator=validator)
    if path == "":
        datasets = load_definitions(joinpath(THISDIR, 'definitions'))
    else:
        datasets = []
    while path != "":
        # look for json files in path
        datasets += load_definitions(path)

        # asking for a new path
        path = prompt(question, completer=path_completer, validator=validator)
    return datasets


def ftp_download(d, credential, install_dir, parsed_url=None):
    """
    NO MORE USED
    download all files at d['install']['url'] using user and password in `credentials`
    """
    if parsed_url is None:
        parsed_url = urlparse(d['install']['url'])
    os.makedirs(os.path.join(install_dir, d['name']), exist_ok=True)
    downloaded_files = []

    # ftp
    with FTP(parsed_url.netloc) as ftp:
        ftp.login(user=credential['user'], passwd=credential['passwd'])
        ftp.cwd(parsed_url.path)
        filenames = ftp.nlst()  # get filenames within the directory

        for filename in filenames:
            local_filename = os.path.join(install_dir, d['name'], filename)
            file = open(local_filename, 'wb')
            ftp.retrbinary('RETR ' + filename, file.write)
            if local_filename.endswith('.zip'):
                downloaded_files.append(local_filename)

    return downloaded_files


def get_credentials(data):
    """
    NO MORE USED
    """
    credentials = [d['name'] for d in data if d['install']['login']]
    for i, credential in enumerate(credentials):
        print("Login credentials for " + credential)
        user = input("User: ")
        password = getpass("Password: ")
        credentials[i] = {"user": user, "passwd": password}

    print()
    return credentials


def intro(data):
    f = Figlet(font='standard')
    print(f.renderText('Audio\nScore\nMeta'))
    f = Figlet(font='sblood')
    print(f.renderText('Dataset'))
    print()

    print("Author: " + data['author'])
    print("Year: ", data['year'])
    print("Website: ", data['url'])


def download(item, credentials, install_dir):
    """
    Really download the files. Credentials (from login) are supported only for
    FTP connections for now.
    """
    # getting credential credentials
    if item['install']['login']:
        credential = credentials.popleft()

    # getting the protocol and the resource to be downloaded
    parsed_url = urlparse(item['install']['url'])
    if parsed_url.scheme == 'ftp':
        # FTP
        # at now, no FTP connection is needed
        downloaded_files = ftp_download(item, credential, install_dir,
                                        parsed_url)
    elif parsed_url.netloc == "mega.nz":
        # mega, using mega.py module
        print("Downloading from mega.nz...")
        mega = Mega()
        downloaded_files = [mega.download_url(item['install']['url'])]
    else:
        # http, https
        with alive_bar(unknown='notes2', spinner='notes_scrolling') as bar:
            temp_fn, _header = urlretrieve(item['install']['url'],
                                           filename=os.path.join(
                                               install_dir, 'temp'),
                                           reporthook=lambda x, y, z: bar)
        downloaded_files = [temp_fn]
    return downloaded_files


def chose_install_dir(json_file):
    validator = Validator.from_callable(
        lambda x: os.path.isdir(x) or x == '',
        error_message="Not a directory!",
        move_cursor_to_end=True,
    )
    path_completer = PathCompleter(only_directories=True)

    default_dir = json_file['install_dir'] or './'

    question = "\nPath to install datasets [empty to default " + \
        default_dir + "] "
    install_dir = prompt(question,
                         validator=validator,
                         completer=path_completer)
    if not install_dir:
        if not os.path.isdir(default_dir):
            os.mkdir(default_dir)
        install_dir = default_dir

    if install_dir.endswith('/'):
        install_dir = install_dir[:-1]

    json_file['install_dir'] = install_dir
    return install_dir


def main():

    with open(joinpath(THISDIR, 'datasets.json')) as f:
        json_file = json.load(f)

    intro(json_file)

    f = Figlet(font='digital')
    print(f.renderText("\nInitial setup"))
    install_dir = chose_install_dir(json_file)
    data = definitions_path()
    print(f.renderText("\nChosing datasets"))
    chose_dataset(data, install_dir)

    # at now, no credential is needed
    credentials = deque(get_credentials(data))
    print(f.renderText("\nProcessing"))
    for d in data:
        full_path = os.path.join(install_dir, d['name'])
        print("Creating " + d['name'])

        if d['install']['url'] != 'unknown':
            print("Downloading (this can take a looooooooooooooot)...")
            downloaded_file = download(d, credentials, install_dir)

        # unzipping if needed
        if d['install']['unpack']:
            print("Unpacking downloaded archive...")
            for temp_fn in downloaded_file:
                format = ''.join(pathlib.Path(
                    d['install']['url']).suffixes) or '.zip'
                format = [
                    j for i, j in supported_archives.items()
                    if format.endswith(i)
                ][0]
                unpack_archive(temp_fn, full_path, format)
                # cleaning up
                os.remove(temp_fn)

        # post-processing
        if d['install']['post-process'] != 'unknown':
            # the following line is only for POSIX!!!
            print("Post-processing (this could take a biiiiiiiiiiiiiiit)...")
            # recursively concatenate commands
            command = '; '.join(d['install']['post-process'])
            command = command.replace('&install_dir', install_dir)
            command = command.replace('&asmd_dir', THISDIR)

            # writing commands to temporary file and executing it as a shell
            # script
            with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tf:
                tf.write(command)
                tf_name = tf.name
                p = Popen(['/bin/env', 'sh', tf_name],
                          stdout=DEVNULL,
                          stderr=DEVNULL)

            # progress bar while script runs
            with alive_bar(unknown='notes_scrolling', spinner='notes') as bar:
                while p.poll() is None:
                    bar()
                    time.sleep(1)

            # removing script
            os.remove(tf_name)

        # just to be sure
        urlcleanup()

    # downloading ground-truth and unpacking
    if confirm("Do you want to download the annotations (about 250MB)?"):
        print(f.renderText("\nDownloading archive from mega.nz..."))
        gt_archive_fn = 'ground_truth.tar.gz'
        # gt_archive_fn = joinpath(THISDIR, 'ground_truth.tar.gz')
        mega = Mega()
        mega.download_url(LINK_GROUND_TRUTH, dest_filename=gt_archive_fn)
        print(f.renderText("\nUnpacking ground-truths"))
        # unpacking the ground_truth data of only the files of the chosen
        # datasets
        with tarfile.open(gt_archive_fn, mode='r:gz') as tf:
            # taking only paths that we really want
            subdir_and_files = [
                member for member in tf.getmembers() for d in data
                if d['name'] in member.name
            ]
            tf.extractall(path=install_dir, members=subdir_and_files)

    # saving the Json file as modified
    with open(joinpath(THISDIR, 'datasets.json'), 'w') as fd:
        json.dump(json_file, fd, indent=4)


if __name__ == '__main__':
    main()
