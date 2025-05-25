import numpy as np

# Example array
arr = np.array([[2.7916371e-05, 9.6902321e-04, 5.4397322e-02, 3.9662010e-04, 4.4260585e-05, 8.5884024e-04, 9.4291294e-01, 3.9306193e-04]])

print(type(arr))

print(arr.shape)

# Find the maximum value in the entire array
max_val = np.max(arr)
print(max_val)

# Find the maximum value along a specific axis (e.g., columns)
max_val_along_columns = np.max(arr, axis=1)
print(max_val_along_columns)

print(np.max(arr))