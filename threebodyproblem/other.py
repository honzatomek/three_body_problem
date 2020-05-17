#!/usr/bin/python3

# <---------------------------------------------------------------------------- general imports --->
import os
import sys
import datetime

# <----------------------------------------------------------------------------- help functions --->
def query_yes_no(question: str='Continue?'):
    prompt = '[?] ' + question + ' [Y/n]: '
    answer = fprint(prompt, question=True)
    if answer.lower() in ['y', 'yes']:
        return True
    elif answer.lower() in ['n', 'no']:
        return False
    else:
        return True

def query_choice(question: str, choices: list):
    prompt = '[?] ' + question + ' ' + str(choices) + ' [' + [c for c in choices[0] if c == c.upper()][0] + ']: '
    answer = fprint(prompt, question=True)
    if answer.lower() in [ch.lower() for ch in choices]:
        return answer.lower()
    else:
        for ch in choices:
            if answer.upper() in ch:
                return ch.lower()
        return False


def timestamp(out='fullname'):
    """
    Returns timestamp, possible outputs: fullname [full, date, datename ,time]
    fullname: 2020-Feb-01 07:28:15
    full:     20200201_072815
    datename: 2020-Feb-01
    date:     2020-02-01
    time:     07:28:15
    """
    if out == 'fullname':
        return '{0:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif out == 'full':
        return '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif out == 'datename':
        return '{0:%Y-%b-%d}'.format(datetime.datetime.now())
    elif out == 'date':
        return '{0:%Y-%m-%d}'.format(datetime.datetime.now())
    elif out == 'time':
        return '{0:%H:%M:%S}'.format(datetime.datetime.now())
    else:
        return '{0:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())

def fprint(prompt: str, question: bool=False, returnstr: bool=False):
    """
    Fancy print function
    """
    tags = {'[o]': '[\033[01;32m+\033[0m]',
            '[ok]': '[\033[01;32m+\033[0m]',
            '[+]': '[\033[01;32m+\033[0m]',
            '[e]': '[\033[01;31m-\033[0m]',
            '[er]': '[\033[01;31m-\033[0m]',
            '[error]': '[\033[01;31m-\033[0m]',
            '[-]': '[\033[01;31m-\033[0m]',
            '[i]': '[\033[01;36mi\033[0m]',
            '[if]': '[\033[01;36mi\033[0m]',
            '[ifo]': '[\033[01;36mi\033[0m]',
            '[info]': '[\033[01;36mi\033[0m]',
            '[q]': '[\033[01;33m?\033[0m]',
            '[qu]': '[\033[01;33m?\033[0m]',
            '[question]': '[\033[01;33m?\033[0m]',
            '[?]': '[\033[01;33m?\033[0m]',
            '[c]': '[\033[01;41m!\033[0m]',
            '[cr]': '[\033[01;41m!\033[0m]',
            '[critical]': '[\033[01;41m!\033[0m]',
            '[!]': '[\033[01;41m!\033[0m]'}

    for tag in tags.keys():
        if prompt.startswith(tag):
            prompt = prompt.replace(tag, tags[tag], 1)
            break
    if returnstr:
        return prompt
    elif question:
        return input(prompt)
    else:
        print(prompt)

if __name__ == '__main__':
    print('testing fprint functionality')
    fprint('[o] [o] ok')
    fprint('[ok] [ok] ok')
    fprint('[+] [+] ok')
    fprint('[e] [e] error')
    fprint('[er] [er] error')
    fprint('[error] [error] error')
    fprint('[-] [-] error')
    fprint('[i] [i] info')
    fprint('[if] [if] info')
    fprint('[info] [info] info')
    fprint('[q] [q] question')
    fprint('[qu] [qu] question')
    fprint('[question] [question] question')
    fprint('[?] [?] question')
    fprint('[c] [c] critical')
    fprint('[cr] [cr] critical')
    fprint('[critical] [critical] critical')
    fprint('[!] [!] critical')


