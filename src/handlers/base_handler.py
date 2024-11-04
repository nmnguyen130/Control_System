from abc import ABC, abstractmethod
import pandas as pd
import os

class BaseCommandHandler(ABC):
    def __init__(self, csv_path=None):
        self.commands = self.load_commands(csv_path) if csv_path else {}
        self.reverse_commands = {value['value']: key for key, value in self.commands.items()}

    @abstractmethod
    def load_commands(self, command_file):
        """Load commands from file - to be implemented by child classes"""
        pass

    def get_value_by_command(self, command):
        """Get numerical value for a command"""
        return self.commands.get(command, {}).get('value', None)

    def get_action_by_command(self, command):
        """Get action for a command"""
        return self.commands.get(command, {}).get('action', None)

    def get_command_by_value(self, value):
        """Get command name from numerical value"""
        return self.reverse_commands.get(value, None)

    def get_all_commands(self):
        """Get all available commands"""
        return self.commands
