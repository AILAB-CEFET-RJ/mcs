from abc import ABC, abstractmethod


class BaseDataSource(ABC):
    @abstractmethod
    def get_data(self, station_id, initial_datetime, final_datetime):
        pass