def time_to_str(time: float) -> str:
    """
    Takes a time in sec and convert to hh:mm:ss
    @param 
        time: time in sec
    @return
        formatted string with hh:mm:ss formatted time
    """
    hours: int = int(time // 3600)
    minutes: int = int((time % 3600) // 60)
    seconds: int = int(time % 60)
    return f"{hours}h:{minutes}m:{seconds}s"