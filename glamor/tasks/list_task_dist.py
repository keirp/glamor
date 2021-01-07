from glamor.tasks.base import BaseTaskDist
import random

class ListTaskDist(BaseTaskDist):

	def __init__(self, tasks):
		self.tasks = tasks
		self._name = 'list_task'

	def __iter__(self):
		self.idx = 0
		return self

	def __next__(self):
		if self.idx < len(self.tasks):
			task = self.tasks[self.idx]
			self.idx += 1
			return task
		else:
			raise StopIteration

class RandomListTaskDist(ListTaskDist):

	def __init__(self, tasks):
		super().__init__(tasks)
		self._name = 'random_list_task'

	def __next__(self):
		return random.choice(self.tasks)