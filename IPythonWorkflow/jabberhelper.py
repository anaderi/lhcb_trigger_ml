__author__ = 'Alex Rogozhnikov'


import sys, os, time
# if you don't have xmpp python module you can either install it
# from egg: http://sourceforge.net/projects/xmpppy/files/
# or just don't use JabberHelper.ipynb - it isn't mandatory component
import xmpp

def SendJabber(to_jid, text):
    """To use this function xmpppy is needed,
    http://xmpppy.sourceforge.net/
    Can be installed on ubuntu with
    apt-get install python-xmpp

    The credentials are kept in the ~/.xsend file
    (will be created automatically after first call)

    Example:
    SendJabber("my@jabber.com", "Analysis is complete, IPython")
    Tested with yandex jabber, seems not to work with google accounts
    (google doesn't support jabber anymore)
    """

    jidparams={}
    if os.access(os.environ['HOME']+'/.xsend',os.R_OK):
        for ln in open(os.environ['HOME']+'/.xsend').readlines():
            if not ln[0] in ('#',';'):
                key,val=ln.strip().split('=',1)
                jidparams[key.lower()]=val
    for mandatory in ['jid','password']:
        if mandatory not in jidparams.keys():
            open(os.environ['HOME']+'/.xsend','w').write(
                    '#Uncomment fields before use and type in correct credentials.\n'
                    + '#JID=romeo@montague.net/resource (/resource is optional)\n#PASSWORD=juliet\n')
            print 'Please fill ~/.xsend config file to valid JID for sending messages.'
            print 'The file is placed: ', os.environ['HOME']+'/.xsend'
            sys.exit(0)

    SendJabberMessageWithPass(jidparams['jid'], jidparams['password'], to_jid, text)


def SendJabberMessageWithPass(from_jid, password, to_jid, text):
    """A much worth way: each time needs a password,
    use SendJabber function instead
    ATTENTION
    Make sure not publish code anywhere if you use this function,
    because your password will be published too.
    """
    print from_jid
    jid = xmpp.protocol.JID(from_jid)
    cl = xmpp.Client(jid.getDomain(), debug=[])

    connection = cl.connect()
    if not connection:
        print 'could not connect!'
        sys.exit()
    print 'connected with', connection
    auth = cl.auth(jid.getNode(), password, resource=jid.getResource())
    if not auth:
        print 'could not authenticate!'
        sys.exit()
    print 'authenticated using', auth

    #cl.SendInitPresence(requestRoster=0)   # you may need to uncomment this for old server
    id = cl.send(xmpp.protocol.Message(to_jid, text))
    #print to_jid, text
    print 'sent message with id', id

    time.sleep(1)   # some older servers will not send the message
                     # if you disconnect immediately after sending

    cl.disconnect()


# Try this with your jabber account
# SendJabber("youraccount@yandex.ru", "Notification")