import win32api

self_info = {
    'birth date' : '8/19/2025'
}

def get_monitors_info():
    monitors_info = []
    monitors = win32api.EnumDisplayMonitors()

    for i, monitor in enumerate(monitors):
        hMonitor, hdcMonitor, rect = monitor
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        monitor_dict = {
            "monitor_number": i + 1,
            "width": width,
            "height": height,
            "left": left,
            "top": top
        }

        monitors_info.append(monitor_dict)

    return monitors_info

def get_all_corners(monitor_info):
    top_left = (monitor_info['left'], monitor_info['top'])
    right_x = monitor_info['left'] + monitor_info['width']
    top_right = (right_x, monitor_info['top'])
    bottom_left = (monitor_info['left'], monitor_info['top'] + monitor_info['height'])
    bottom_right = (right_x, monitor_info['top'] + monitor_info['height'])
    monitor_info['top_left'] = top_left
    monitor_info['top_right'] = top_right
    monitor_info['bottom_left'] = bottom_left
    monitor_info['bottom_right'] = bottom_right
    return monitor_info
all_monitors_info = []
for m in get_monitors_info():
    all_monitors_info.append(get_all_corners(m))
print(all_monitors_info)