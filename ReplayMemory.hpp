#ifndef _REPLAY_MEM_H
#define _REPLAY_MEM_H

#include <vector>

template <class trainingPair>
class ReplayMemory {
public:
    enum FORGETMODES {FIFO, RANDOM_EQUAL, RANDOM_OLDER};
    ReplayMemory() : g(rd()) {
        setSizeLimit(64);
    }
    void setMemoryLimit(const size_t limit) {
        memlimit=limit;
        mem.resize(memlimit);
        //fill index list - later this is shuffled for random samples
        indexList.resize(memlimit);
        for(size_t i=0; i < memlimit; i++) indexList[i]=i;
    }

    void add(trainingPair &tp, size_t timestamp) {
        if (mem.size() == memLimit) {
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
                        size_t threhold = distTS(g);
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

    std::vector<trainingPair> sample(size_t nMemories) {
        std::vector<trainingPair> samp;
        samp.resize(std::min(nMemories, memlimit));
        std::shuffle(indexList.begin(), indexList.end(), g);
        for(size_t i=0; i < samp.size(); i++) {
            samp[i] = mem.at(indexList[i]).tp;
        }
        return samp;
    }
private:
    struct memory {
        size_t timestamp;
        trainingPair tp;
    };
    std::deque<memory> mem;
    size_t memlimit=64;
    std::vector<size_t> indexList;
    std::random_device rd;
    std::mt19937 g;
    FORGETMODES forgettingMode = FORGETMODES::FIFO;
};

#endif

