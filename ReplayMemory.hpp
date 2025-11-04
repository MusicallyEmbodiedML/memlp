/**
 * @file ReplayMemory.hpp
 * @brief Replay Memory implementation for MEMLP Library
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef _REPLAY_MEM_H
#define _REPLAY_MEM_H

#include <vector>
#include <deque>
#include <random>

/**
 * @brief Represents a training sample consisting of input (X) and output (Y) vectors.
 *
 * @tparam T The data type of the elements in the vectors.
 */
template<typename T>
struct trainXYItem {
    std::vector<T> X; /**< Input vector. */
    std::vector<T> Y; /**< Output vector. */
};

/**
 * @brief Implements a replay memory system for storing and sampling training data.
 *
 * This class is designed to manage a fixed-size memory buffer for training items.
 * It supports multiple forgetting modes to handle memory overflow and provides
 * functionality to sample random subsets of stored items.
 *
 * @tparam trainingItem The type of the training items stored in the replay memory.
 */
template <class trainingItem>
class ReplayMemory {
public:
    /**
     * @brief Enumeration of forgetting modes for handling memory overflow.
     */
    enum FORGETMODES {
        FIFO,           /**< First-In-First-Out: Removes the oldest item. */
        RANDOM_EQUAL,   /**< Random Equal: Removes a random item with equal probability. */
        RANDOM_OLDER    /**< Random Older: Removes an older item with higher probability. */
    };

    /**
     * @brief Constructs a ReplayMemory object with a default memory limit of 64 items.
     */
    ReplayMemory() : g(rd()) {
        setMemoryLimit(64);
    }

    /**
     * @brief Sets the maximum number of items that can be stored in the replay memory.
     *
     * @param limit The maximum number of items to store.
     */
    void setMemoryLimit(const size_t limit) {
        memlimit = limit;
    }

    size_t getMemoryLimit() {
        return memlimit;
    }

    /**
     * @brief Adds a new training item to the replay memory.
     *
     * If the memory is full, an existing item is removed based on the current forgetting mode.
     *
     * @param tp The training item to add.
     * @param timestamp The timestamp associated with the training item.
     */
    void add(trainingItem &tp, size_t timestamp) {
        if (mem.size() == memlimit) {
            switch (forgettingMode) {
                case FORGETMODES::FIFO:
                {
                    //remove from front
                    mem.erase(mem.begin());
                    // Serial.println("FIFO: removing oldest item from memory");
                    break;
                }
                case FORGETMODES::RANDOM_EQUAL:
                {
                    std::uniform_int_distribution<size_t> dist(0, memlimit-1);
                    size_t index = dist(g);
                    mem.erase(mem.begin() + index);
                    break;
                }
                case FORGETMODES::RANDOM_OLDER:
                {
                    size_t maxAge = mem.at(0).timestamp;
                    std::uniform_int_distribution<size_t> distTS(0, maxAge * 2);
                    std::uniform_int_distribution<size_t> distIndex(0, memlimit);
                    bool chosen=false;
                    while(!chosen) {
                        size_t index = distIndex(g);
                        size_t threshold = distTS(g);
                        if (threshold < mem.at(index).timestamp) {
                            chosen = true;
                            mem.erase(mem.begin() + index);
                        }
                    }
                    break;
                }
            }
        }
        memory newMemory {timestamp, tp};
        mem.push_back(newMemory);
        // Serial.printf("Adding item to memory, size: %d\n", mem.size());
    }

    /**
     * @brief Samples a random subset of training items from the replay memory.
     *
     * @param nMemories The number of items to sample.
     * @return A vector containing the sampled training items.
     */
    std::vector<trainingItem> sample(size_t nMemories) {
        std::vector<trainingItem> samp;
        samp.resize(std::min(nMemories, mem.size()));

        if (mem.size() > 0) {
            // std::shuffle(indexList.begin(), indexList.end(), g);
            //risk of identical items in selection, but much quicker on an MCU
            std::uniform_int_distribution<size_t> dist(0, mem.size()-1);

            for(size_t i=0; i < samp.size(); i++) {
                size_t index = dist(g);
                samp[i] = mem.at(index).item;
            }
        }
        return samp;
    }

    std::vector<size_t> sampleIndices(size_t nMemories) {
	    std::vector<size_t> indices;
		indices.reserve(std::min(nMemories, mem.size()));

		if (mem.size() > 0) {
			std::uniform_int_distribution<size_t> dist(0, mem.size()-1);

			for(size_t i=0; i < std::min(nMemories, mem.size()); i++) {
				indices.push_back(dist(g));
			}
		}
		return indices;
	}

	/**
	 * @brief Returns a reference to an item at the specified index for in-place editing.
	 * 
	 * @param index The index of the item to access.
	 * @return A reference to the training item.
	 */
	inline trainingItem& getItem(size_t index) {
		return mem.at(index).item;
	}

	// Optional: const version for read-only access
	const trainingItem& getItem(size_t index) const {
		return mem.at(index).item;
	}

    void eraseItem(const size_t index) {
        mem.erase(mem.begin() + index);
    }
		
// Then access with: mem[index].item
    void clear() {
        mem.clear();
    }

    /**
     * @brief Removes multiple items by their indices efficiently.
     * 
     * @param indicesToRemove Vector of indices to remove (will be sorted internally).
     */
    bool removeItems(std::vector<size_t> indicesToRemove) {
        if (indicesToRemove.empty()) return false;
        
        // Sort in descending order to remove from back to front
        std::sort(indicesToRemove.begin(), indicesToRemove.end(), std::greater<size_t>());
        
        // Remove duplicates
        indicesToRemove.erase(std::unique(indicesToRemove.begin(), indicesToRemove.end()), 
                            indicesToRemove.end());
        
        // Remove items from back to front so indices remain valid
        for (size_t idx : indicesToRemove) {
            if (idx < mem.size()) {
                mem.erase(mem.begin() + idx);
            }
        }
        return true;
    }    

private:
    /**
     * @brief Represents a memory entry containing a training item and its timestamp.
     */
    struct memory {
        size_t timestamp; /**< The timestamp of the training item. */
        trainingItem item; /**< The training item stored in memory. */
    };

    std::deque<memory> mem; /**< The deque storing memory entries. */
    size_t memlimit = 64; /**< The maximum number of items in the replay memory. */
    std::vector<size_t> indexList; /**< List of indices for sampling. */
    std::random_device rd; /**< Random device for generating random numbers. */
    std::mt19937 g; /**< Mersenne Twister random number generator. */
    FORGETMODES forgettingMode = FORGETMODES::FIFO; /**< The current forgetting mode. */
};

#endif
