import sys
import urllib.request
from slack_webhook import Slack
#import socket
#import psutil
import GPUtil
import subprocess

def slack_status():
        #proc1=subprocess.Popen('python -c "import GPUtil;GPUtil.showUtilization()"', shell=True, stdout=subprocess.PIPE, )
        proc1=subprocess.Popen('gpustat', shell=True, stdout=subprocess.PIPE, )
        proc2=subprocess.Popen('tail -1 /root/recursion/nohup.out', shell=True, stdout=subprocess.PIPE, )
        gpu_stats =proc1.communicate()[0].decode("utf-8")
        training_status = proc2.communicate()[0].decode("utf-8")
        #gives an object with many fields
        #psutil.virtual_memory()
        #you can convert that object to a dictionary pass to slack required values
        #dict(psutil.virtual_memory()._asdict())
        #instanceid = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id').read().decode()
        #replace with the final hook once ready
        #slack = Slack(url='https://hooks.slack.com/services/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')
        slack = Slack(url='https://hooks.slack.com/services/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')
        slack.post(text= training_status, #gpu_stats, #"EC2 Notification | Instance Status",
                      attachments = [{
                              "fallback": "",
                              "author_name": "",
                              "title": "",
                              "text":  "",
                              "actions": []
                      }])


# + socket.gethostname(),
if __name__ == "__main__":
    #slack_status(sys.argv[1])
    slack_status()
	
#call this script from you training code like this:
#slack_status("add_training_data_as_string_here")

#or from Bash if it croned like this:
# #!/bin/bash
# python3 web_hook.py "sometraininginfo"