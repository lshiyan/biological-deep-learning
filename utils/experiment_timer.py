def time_to_str(time_in_second: float) -> str:
    """
    FUNCTION
    Takes a time in sec and convert to hh:mm:ss
    @param 
        time_in_sec: time in sec
    @return
        formatted string with hh:mm:ss formatted time
    """
    hours: int = int(time_in_second // 3600)
    minutes: int = int((time_in_second % 3600) // 60)
    seconds: int = int(time_in_second % 60)
    return f"{hours}h:{minutes}m:{seconds}s"