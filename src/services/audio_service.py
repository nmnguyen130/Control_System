from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER

class AudioControl:
    def __init__(self):
        # Lấy thiết bị âm thanh mặc định (loa)
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, 
            1,  # Multimedia device role
            None
        )
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

    def increase_volume(self):
        """Tăng âm lượng"""
        current_volume = self.volume.GetMasterVolumeLevelScalar()
        if current_volume < 1.0:
            self.volume.SetMasterVolumeLevelScalar(current_volume + 0.1, None)
            return "Volume increased"
        return "Volume is already at maximum"

    def decrease_volume(self):
        """Giảm âm lượng"""
        current_volume = self.volume.GetMasterVolumeLevelScalar()
        if current_volume > 0.0:
            self.volume.SetMasterVolumeLevelScalar(current_volume - 0.1, None)
            return "Volume decreased"
        return "Volume is already at minimum"

    def mute(self):
        """Tắt tiếng"""
        self.volume.SetMute(True, None)
        return "Volume muted"

    def unmute(self):
        """Bỏ tắt tiếng"""
        self.volume.SetMute(False, None)
        return "Volume unmuted"