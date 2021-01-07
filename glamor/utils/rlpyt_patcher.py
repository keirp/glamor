import rlpyt.utils.logging.logger as logger
import wandb

def _patched_dump_tabular(*args, **kwargs):
	for line in logger.tabulate(logger._tabular).split('\n'):
		logger.log(line, *args, **kwargs)
	tabular_dict = dict(logger._tabular)
	clean_tabular_dict = {}
	for key in tabular_dict:
		parts = key.split('/')
		for i in range(len(parts)):
			parts[i] = ''.join(e for e in parts[i] if e.isalnum())
		clean_key = f'rlpyt/{parts[-1]}'
		clean_tabular_dict[clean_key] = float(tabular_dict[key])
	wandb.log(clean_tabular_dict)
	del logger._tabular[:]

def patch_logger():
	logger.dump_tabular = _patched_dump_tabular
