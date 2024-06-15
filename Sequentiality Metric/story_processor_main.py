"""
This is the main File for execution. We can specify the input file and history to be run for
"""

import pandas as pd
from LLM import LLMInference


class StoryProcessor:

    def __init__(self, input_file):
        self.input_file_name = input_file
        self.llm_model = LLMInference()
        self.sequentiality = []
        self.topic_output = []
        self.contextual_output = []
        self.output_file_name = f"{input_file}_result.csv"

    def compute_sequentiality_for_file(self, index_range=None, previous_history_size=None, model_name="gpt2"):
        """

        :param index_range: For Debug purposes we can specify index range for execution
        :param previous_history_size: History size for sequentiality metric
        :param model_name: LLM Model to be used
        :return: Computes Sequentiality and writes it to File
        """

        story_data_frame = pd.read_csv(self.input_file_name)
        try:

            if index_range:
                story_data_frame = story_data_frame.iloc[index_range[0]: index_range[1]]
            for index, row in story_data_frame.iterrows():
                print(f"Processing story row {self.output_file_name} {index + 1} / {len(story_data_frame)}")
                topic_column = row['summary']
                story = row['story']
                result = self.llm_model.compute_story_metric(story, topic_column, history_size=previous_history_size,
                                                             model=model_name)
                self.sequentiality.append(result[0])
                self.topic_output.append(result[1])
                self.contextual_output.append(result[2])
        except Exception as e:
            print("Exception occured: ", e)
        finally:
            story_data_frame['c_value'] = self.sequentiality
            story_data_frame['topic_output'] = self.topic_output
            story_data_frame['contextual'] = self.contextual_output

            story_data_frame.to_csv(self.output_file_name)


if __name__ == "__main__":
    StoryProcessor(
        "/Users/pranoysarath/PycharmProjects/pythonProject1/data/filtered/combined_filtered_8.csv").compute_sequentiality_for_file(
        previous_history_size=9)
