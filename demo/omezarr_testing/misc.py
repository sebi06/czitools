# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Author      : SRh, JSm
# Institution : Carl Zeiss Microscopy GmbH
#
#
# Copyright(c) 2025 Carl Zeiss AG, Germany. All Rights Reserved.
#
# Permission is granted to use, modify and distribute this code,
# as long as this copyright notice remains part of the code.
#################################################################


import configparser
import ssl
from grpclib.client import Channel
from typing import List, Tuple
from pathlib import Path

from pydantic import ValidationError
from loguru import logger
import sys


def set_logging():
    """
    Configures the logging settings for the application.
    This function removes any existing loggers and sets up a new logger
    that outputs to the standard output stream (sys.stdout). The log
    messages are colorized and formatted to include the timestamp, log
    level, and message.
    Returns:
        logger (Logger): The configured logger instance.
    """

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}s</green> - <level>{level}</level> - <level>{message}</level>",
    )

    return logger


def initialize_zenapi(
    config_file: str = "config.ini",
) -> Tuple[Channel, List[Tuple[str, str]]]:
    """
    Create an gRPC channel for ZEN-API using the specified configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Tuple consisting of gRPC Channel and metadata as a list of tuples.
    """

    # get the configuration
    config = configparser.ConfigParser()
    config.read(config_file)

    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # Load a “certification authority” (CA) certificate
    context.load_verify_locations(cafile=config["api"]["cert_file"])
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True

    # Set protocol which should be used during the SSL/TLS handshake
    context.set_alpn_protocols(["h2"])
    channel = Channel(host=config["api"]["host"], port=int(config["api"]["port"]), ssl=context)
    metadata = [("control-token", config["api"]["control-token"])]

    return channel, metadata


def path_validator(v):
    """Returns a Path object.

    Args:
        v: A string or Path object.

    Returns:
        A Path object.

    Raises:
        ValidationError: If the value is not a valid path.
    """
    if isinstance(v, (str, Path)):
        return Path(v)
    raise ValidationError("value is not a valid path")
