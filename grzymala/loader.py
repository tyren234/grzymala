import time
import threading
from itertools import cycle

class LoadingAnimation:
    def __init__(self, interval : "float" = 0.2, animation : "list[str]" = [
"[=     ]",
"[ =    ]",
"[  =   ]",
"[   =  ]",
"[    = ]",
"[     =]",
"[    = ]",
"[   =  ]",
"[  =   ]",
"[ =    ]",
],
finishMessage : "str" = "",
failedMessage : "str" = "",
):
        self.animation = animation
        self.interval = interval
        self.animating = False
        self.thread = threading.Thread(group=None, target=self.animate)
        self.finishMessage = finishMessage
        self.failedMessage = failedMessage

    def animate(self):
        for c in cycle(self.animation):
            if not self.animating:
                break
            print(c, end="\r")
            time.sleep(self.interval)
    
    def start(self):
        self.animating = True
        self.thread.start()
    
    def stop(self):
        if self.animating:
            self.animating = False
            print(self.finishMessage)
    
    def fail(self):
        if self.animating:
            self.animating = False
            print(self.failedMessage)