import socket
import time

# HOST = '192.168.0.46'
# PORT = 8088

# HOST = '192.168.0.162'
# PORT = 8088

HOST = '10.0.0.230'
PORT = 8088

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')

while True:
    conn, addr = s.accept()
    print('connected by ' + str(addr))
    while True:
        # for i in range(100): 
        #     # data = input('Input data to send:')
        #     outdata = str(i)
        #     conn.send(outdata.encode())
        #     time.sleep(0.5)
        conn.send('signal from chiou'.encode())
        time.sleep(2)

    # s.close()
    # break
