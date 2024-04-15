names = (3, 5, 6, 7, 19, 21, 44, 47, 22, 51, 53, 64)
labels = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)
lines = ['Video Name,Label\n']
for name, label in zip(names, labels):
    for i in range(6):
        lines.append(f'{name}_{i},{label}\n')


with open('label.csv', 'w') as f:
    f.writelines(lines)