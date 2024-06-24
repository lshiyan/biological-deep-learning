def time_to_str(time: float) -> str:
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    return f"{hours}h:{minutes}m:{seconds}s"