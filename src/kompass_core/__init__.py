import kompass_cpp


def set_logging_level(level: str):
    """Set the logging level to "debug", "warn", "error" or "info"

    :param level: Logging level
    :type level: str
    """
    level = level.upper()
    kompass_cpp.set_log_level(getattr(kompass_cpp.LogLevel, level))


__all__ = ["set_logging_level"]
