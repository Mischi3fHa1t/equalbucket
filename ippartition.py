"""
Split up an array of IP address to number of near equal partitions
Time Complexity: O(nlogn)
Space Complexity: O(n)
"""
import ipaddress

def ip_to_int(ipaddr):
    """Convert an IP address to an integer."""
    return int(ipaddress.IPv4Address(ipaddr))

def int_to_ip(num):
    """Convert an integer to an IP address."""
    return str(ipaddress.IPv4Address(num))

def partition_ips(ipaddrs, num_partitions):
    """Partition IP addresses into the specified number of partitions."""
    # Convert IP addresses to integers and sort them
    ip_integers = sorted(ip_to_int(ip) for ip in ipaddrs)
    total_ips = len(ip_integers)

    # Calculate the size of each partition
    partition_size = total_ips // num_partitions
    partitions = []

    for i in range(num_partitions):
        start_index = i * partition_size
        if i == num_partitions - 1:
            end_index = total_ips  # Include all remaining IPs in the last partition
        else:
            end_index = (i + 1) * partition_size

        partitions.append(ip_integers[start_index:end_index])

    return partitions

def assign_tags(partitions, tags):
    """Assign tags to each partition."""
    tagged_partitions = []
    for i, partition in enumerate(partitions):
        tagged_partitions.append((tags[i], [int_to_ip(ip) for ip in partition]))
    return tagged_partitions

def main(ip_list):
    """
    test 4 partitions with 8 ips
    """
    tags = ['n1', 'n2', 'n3', 'n4']
    num_partitions = len(tags)

    partitions = partition_ips(ip_list, num_partitions)
    tagged_partitions1 = assign_tags(partitions, tags)

    return tagged_partitions1

# Example usage
ip_list1 = [
    '10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4', 
    '10.0.0.5', '10.0.0.6', '10.0.0.7', '10.0.0.8',
    # Add more IP addresses as needed
]

tagged_partitions2 = main(ip_list1)
for tag, ips in tagged_partitions2:
    print(f"Tag {tag}:")
    for ip in ips:
        print(f"  {ip}")
