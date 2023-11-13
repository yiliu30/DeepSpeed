import torch

# Create a sample PyTorch parameter (you would have your own parameter)
value = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define values for the number of partitions and rank
num_partitions = 2
my_rank = 0

# Calculate the length of each partition
partition_length = value.numel() // num_partitions

# Perform the flattening and narrowing operations
flattened_value = value.flatten()
start_index = partition_length * my_rank
narrowed_value = flattened_value.narrow(0, start_index, partition_length)

# Print the results
print("Original value:")
print(value)

print("Flattened value:")
print(flattened_value)

print("Narrowed value:")
print(narrowed_value)




# Original value:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
# Flattened value:
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# rank 0
# Narrowed value:
# tensor([1, 2, 3, 4])

# rank 1
# Narrowed value:
# tensor([5, 6, 7, 8])
