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
        // mem.resize(memlimit);
        //fill index list - later this is shuffled for random samples
        // indexList.resize(memlimit);
        // for(size_t i=0; i < memlimit; i++) indexList[i]=i;
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
        // std::shuffle(indexList.begin(), indexList.end(), g);
        //risk of identical items in selection, but much quicker on an MCU
        std::uniform_int_distribution<size_t> dist(0, mem.size()-1);

        for(size_t i=0; i < samp.size(); i++) {
            size_t index = dist(g);
            samp[i] = mem.at(index).item;
        }
        return samp;
    }

    void clear() {
        mem.clear();
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
