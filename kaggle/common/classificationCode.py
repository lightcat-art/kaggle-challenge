from enum import Enum


class TaskMap(Enum):
    """
    TASK_NAME = {'MODEL_TYPE_PARAM_NAME':'MODEL_TYPE_FOLDER_NAME'}
    """
    PATENTMATCHING = {'MSE': 'MSE'}


if __name__ == "__main__":
    print(TaskMap.PATENTMATCHING.value.get('MSE'))
    print(TaskMap.PATENTMATCHING.name)
    pass
