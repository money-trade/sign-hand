"""ค้นหาอุปกรณ์กล้องที่ต่อกับเครื่อง"""
import cv2


def list_cameras(max_probe=8):
    """คืนค่า list ของ (index, ชื่ออุปกรณ์)

    บน Windows ใช้ DirectShow (pygrabber) อ่านชื่อจริงของกล้อง ลำดับตรงกับ index
    ของ cv2.CAP_DSHOW ถ้าไม่มี pygrabber จะลองเปิดกล้องทีละ index แทน
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return list(enumerate(names))
    except Exception:
        pass

    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append((i, f"กล้อง {i}"))
        cap.release()
    return found
