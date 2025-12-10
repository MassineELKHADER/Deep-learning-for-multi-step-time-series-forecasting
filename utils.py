def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


# print(format_time(6949)) # 1h 55m 49s
# print(format_time(4877)) # 1h 21m 17s