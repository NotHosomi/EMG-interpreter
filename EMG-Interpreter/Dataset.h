#pragma once
#include <vector>

template <typename T>
struct Dataset
{
	std::vector<T> inputs;
	std::vector<T> labels;

	void shuffle()
	{
		size_t size = inputs.size();
		for (size_t i = 0; i < size - 1; i++)
		{
			size_t j = i + rand() % (size - i);
			swap(inputs[i], inputs[j]);
			swap(labels[i], labels[j]);
		}
	}

	Dataset<T> split(double percentage)
	{
		std::size_t const split_size = static_cast<size_t>(inputs.size() * percentage);

		std::vector<T> inputs_split(inputs.begin() + split_size, inputs.end());
		std::vector<T> labels_split(labels.begin() + split_size, labels.end());

		inputs.erase(inputs.begin() + split_size, inputs.end());
		labels.erase(labels.begin() + split_size, labels.end());

		Dataset<T> new_split;
		new_split.inputs = inputs_split;
		new_split.labels = labels_split;
		return new_split;
	}
};
