from enum import Enum


class TaskMap(Enum):
    """
    TASK_NAME = {'MODEL_TYPE_NAME':'MODEL_TYPE'}
    """
    PATENTMATCHING = {'MSE': 'MSE'}


if __name__ == "__main__":
    print(TaskMap.PATENTMATCHING.value.get('MSE'))
    print(TaskMap.PATENTMATCHING.name)
    pass
