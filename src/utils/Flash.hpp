#ifndef __FLASH_HPP__
#define __FLASH_HPP__

extern "C" {
#include <quadflashlib.h>
#include <xcore/port.h>
#include <xcore/clock.h>
#include <platform.h>
#include <xs1.h>
}
#include <string>
#include <vector>


class XMOSFlash {

 public:

    XMOSFlash() {
        flashport = {
            PORT_SQI_CS,
            PORT_SQI_SCLK,
            PORT_SQI_SIO,
            XS1_CLKBLK_1
        };

    }

    int connect() {
        return fl_connect(&flashport);
    }

    int disconnect() {
        return fl_disconnect();
    }

    void SetPayload(const std::vector<uint8_t> &data) {
        payload_ = data;
    }

    void GetPayload(std::vector<uint8_t> &data) {
        data = payload_;
    }

    std::vector<uint8_t>* GetPayloadPtr() {
        return &payload_;
    }

    void WriteToFlash() {

        size_t datalen = payload_.size();
        size_t w_head = 0;  // Write head

        // Write in trailer segment
        w_head = _LowLevelWrite(w_head, leader_);
        // Write in amount of data in bytes
        w_head = _LowLevelWrite(
            w_head,
            reinterpret_cast<uint8_t*>(&datalen),
            sizeof(size_t));
        // Write actual data
        w_head = _LowLevelWrite(w_head, payload_);

        printf("FLASH- Written %d bytes, head at %d.\n", datalen, w_head);
    }

    void ReadFromFlash() {

        size_t datalen;
        size_t r_head = 0;

        // Get leader segment
        std::vector<uint8_t> leader;
        r_head = _LowLevelRead(r_head, leader_.size(), leader);
        if (leader != leader_) {
            printf("FLASH- Leader bytes not found, can't load data.\n");
            return;
        }

        // Get actual amount of data
        r_head = _LowLevelReadValue(r_head, datalen);
        printf("FLASH- Data length: %d bytes\n", datalen);
        if (datalen > 0xffff) {
            printf("FLASH- Data length absurd, flash probably dirty. Aborting\n");
            return;
        }

        // Get payload
        r_head = _LowLevelRead(r_head, datalen, payload_);
    }

    void PrintFlashInfo() {
        //get info
        size_t dataPartitionSize = fl_getDataPartitionSize();
        printf("Data partition size: %x\n", dataPartitionSize);
        size_t numDataSectors = fl_getNumDataSectors();
        printf("# data sectors: %x\n", numDataSectors);
        size_t pageSize = fl_getPageSize();
        printf("page size: %x\n", pageSize);
        size_t numDataPages = fl_getNumDataPages();
        printf("# data pages: %x\n", numDataPages);
        // size_t dataSectorSize = fl_getDataSectorSize();
        // printf("Data sector size: %x\n", dataSectorSize);
    }

 protected:

    const std::vector<uint8_t> leader_
            {'M','E','M','L','D','A','T','A'};
    std::vector<uint8_t> payload_;
    fl_QSPIPorts flashport;


    size_t _LowLevelWrite(size_t w_head, const std::vector<uint8_t> &payload) {

        return _LowLevelWrite(w_head, payload.data(), payload.size());
    }

    size_t _LowLevelWrite(size_t w_head, const uint8_t *payload, const size_t payload_size) {

        size_t scratchSize = fl_getWriteScratchSize(w_head, payload_size);
        std::vector<unsigned char> scratch(scratchSize);
        fl_writeData(w_head, payload_size, payload, scratch.data());

        return w_head + payload_size;
    }

    size_t _LowLevelRead(size_t r_head, size_t size, std::vector<uint8_t> &payload) {
        payload.resize(size);
        fl_readData(r_head, size, payload.data());

        return r_head + size;
    }

    template< typename T >
    size_t _LowLevelReadValue(size_t r_head, T &value) {

        union {
            T val;
            uint8_t bytes[sizeof(T)];
        } val_union;
        fl_readData(r_head, sizeof(T), val_union.bytes);
        value = val_union.val;

        return r_head + sizeof(T);
    }

};


static XMOSFlash gFlash;

#endif