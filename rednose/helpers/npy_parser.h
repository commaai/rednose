/*\
 * The npy files are pickled numpy arrays. The first 128 bytes have file
 * metadata in python dictionary repr format, in ASCII encoding, where the key
 * "descr" hints at the number encoding ("<i8" for integers and "<f8" for
 * doubles), and the key "shape" has the shape of the tensor.
 * Each number (array element) is 8-bytes long.
 * Numbers are encoded little-endian.
 * Integers are stored with the two's-complement system, doubles with IEEE 754
 * encoding.
 * To read:
 * 1. Read first 128 bytes as ASCII,
 * 1.1. confirm head reads ".NUMPY" (where . is 0x93),
 * 1.2. parse dict (get number encoding and shape).
 * 2. According to shape, read all numbers starting from byte 128, onwards.
 *    Decode numbers according to encoding indicated by the descr. dict key.
\*/

#include <iostream>
#include <fstream>
#include <cstring>

#include <endian.h>

#include <eigen3/Eigen/Dense>

#define NPYPARSER_HEADER_ERROR 100
#define NPYPARSER_INDEX_ERROR  101

typedef struct Metadata {
    char     num_type; // 0: unspecified, 1: integer, 2: double
    uint64_t rows;
    uint64_t cols;
} Metadata;

double decode_integer(const char *string);

double decode_double(const char *string);

bool check_header(std::ifstream *fobj);

void populate_metadata(std::ifstream  *fobj,
                       Metadata       *metadata);

void populate_lookup_row(std::ifstream   *lut_fobj,
                         Eigen::VectorXd *row,
                         const uint16_t   irow);
