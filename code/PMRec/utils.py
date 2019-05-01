#!/usr/bin/env python


import logging, logging.config


MINIMAL_DEBUG = 'minimal_debug'


# set standard logging configurations
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'minimal': {
            'format': '%(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'minimal': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'minimal'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
        MINIMAL_DEBUG: {
            'handlers': ['minimal'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
})


def make_logger(name):
    return logging.getLogger(name)
