import psutil
import time

# Function to get the current network usage
def get_network_usage():
    net_io = psutil.net_io_counters()
    return net_io.bytes_sent, net_io.bytes_recv

# Get initial network usage
start_sent, start_recv = get_network_usage()

# Run your script or code here
# Replace this with the code that you want to measure bandwidth usage for
time.sleep(1)  # Simulating script running for 10 seconds

# Get final network usage
end_sent, end_recv = get_network_usage()

# Calculate the difference in bytes
bytes_sent = end_sent - start_sent
bytes_recv = end_recv - start_recv

# Convert bytes to megabytes
megabytes_sent = bytes_sent / (1024 * 1024)
megabytes_recv = bytes_recv / (1024 * 1024)

print(f"Sent: {bytes_sent / (1024 * 1024):.2f} MB")  # convert to MB
print(f"Received: {bytes_recv / (1024 * 1024):.2f} MB")  # convert to MB
