import pandas as pd

class CommandHandler:
    def __init__(self, csv_path):
        self.commands = self.load_commands(csv_path)

    def load_commands(self, csv_path):
        """Load commands from a CSV file."""
        if not csv_path:
            return {0: 'nothing'}
        df = pd.read_csv(csv_path)
        return {
            index: row['command'] for index, row in df.iterrows()
        }

    def get_command_index(self, command):
        """Get command index by command."""
        for index, cmd in self.commands.items():
            if cmd == command:
                return index
        return None

    def get_command(self, index):
        """Get command by index."""
        return self.commands.get(index, None)

    def get_all_commands(self):
        """Get all available commands."""
        return self.commands

    def get_command_length(self):
        return len(self.commands)