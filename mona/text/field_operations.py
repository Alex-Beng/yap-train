import random

field_operations_names = [
    f'尚需生长时间：{day}天{hour}小时' for day in range(1, 3) for hour in range(1, 24)
]
field_operations_names += [
    f'尚需生长时间：{minu}分钟' for minu in range(0, 60)
]

def random_field_operation_name():
    return random.choice(field_operations_names)