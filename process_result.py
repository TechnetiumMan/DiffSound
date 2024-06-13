file_path = "out/turtle_thickness/result.txt"
f = open(file_path)
lines = f.readlines()
error_sum = 0
for line in lines:
    if line[0:2] == 'ta':
        target = float(line[7:10])
        if target in [0.3, 0.4, 0.5, 0.6, 0.7]:
            result = float(line[18:])
            error_sum += abs(result - target)
            
print(error_sum / 5)