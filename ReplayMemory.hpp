#ifndef _REPLAY_MEM_H
#define _REPLAY_MEM_H

#include <vector>
#include <deque>
#include <random>

template<typename T>
struct trainXYItem {
    std::vector<T> X;
    std::vector<T> Y;
};

template <class trainingItem>
class ReplayMemory {
public:
    enum FORGETMODES {FIFO, RANDOM_EQUAL, RANDOM_OLDER};
    ReplayMemory() : g(rd()) {
        setMemoryLimit(64);
    }
    void setMemoryLimit(const size_t limit) {
        memlimit=limit;
        // mem.resize(memlimit);
        //fill index list - later this is shuffled for random samples
        // indexList.resize(memlimit);
        // for(size_t i=0; i < memlimit; i++) indexList[i]=i;
    }

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
private:
    struct memory {
        size_t timestamp;
        trainingItem item;
    };
    std::deque<memory> mem;
    size_t memlimit=64;
    std::vector<size_t> indexList;
    std::random_device rd;
    std::mt19937 g;
    FORGETMODES forgettingMode = FORGETMODES::FIFO;
};

#endif

