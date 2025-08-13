#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : logging.py

import datetime
import os


class LogManager:
    def __init__(self,
                 log_to_console: bool = True,
                 path: str = None,
                 *,
                 encoding: str = "utf-8"):
        self.ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if path:
            self._path: str = self.__expend_log(path)
        else:
            home_dir = os.path.expanduser("~")
            log_dir = os.path.join(home_dir, ".deepecon", "log")
            # make sure the _path is existed
            os.makedirs(log_dir, exist_ok=True)
            self._path: str = os.path.join(log_dir, f"{self.ts}.log")

        self._fh = open(self._path, "w", encoding=encoding)
        self.__console = log_to_console

    @staticmethod
    def __expend_log(path) -> str:
        """
        Ensures the log file _path is properly formatted and the directory exists.

        If the _path does not end with '.log', it appends '.log' to the _path.
        Expands the user directory if '~' is present in the _path.
        Creates the directory structure if it doesn't already exist.
        """
        if not path.endswith(".log"):
            path += ".log"
        path = os.path.expanduser(path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        return path

    def write(self, command: str, result: str):
        content = f"· {command}\n>>> {result}\n"
        if self.__console:
            print(content)
        self._fh.write(content)
        self._fh.flush()  # Force write the change to disk

    def _close_file(self):
        if hasattr(self, "_fh") and not self._fh.closed:
            try:
                self._fh.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def rm(self):
        os.remove(self._path)

    def exit_with_save(self,
                       is_save: bool = False,
                       path: str = None):
        try:
            if is_save:
                if path:
                    path = self.__expend_log(path)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    os.replace(self._path, path)
                    self._path = path
            else:
                try:
                    os.remove(self._path)
                except FileNotFoundError:
                    pass
        finally:
            self._close_file()

        return self._path if is_save else None

    def get_path(self):
        return self._path

    def cat_log(self):
        os.system(f"cat {self._path}")


if __name__ == "__main__":
    log = LogManager()
    print(log._path)
    log.write("mock command", "mock result")
    file_path = log.exit_with_save(True)
    print(file_path)
