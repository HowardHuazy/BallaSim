#include "scene/assets/png.h"

#include <png.h>

#include <csetjmp>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "scene/errors.h"

namespace balla::scene::assets {

namespace {

struct File {
  std::FILE* f = nullptr;
  ~File() {
    if (f) std::fclose(f);
  }
};

// libpng's read structs, destroyed on every path out (an exception included).
struct Reader {
  png_structp png = nullptr;
  png_infop info = nullptr;
  ~Reader() {
    if (png) png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
  }
};

thread_local char g_message[256];

[[noreturn]] void on_error(png_structp p, png_const_charp msg) {
  std::snprintf(g_message, sizeof(g_message), "%s", msg);
  png_longjmp(p, 1);
}

void on_warning(png_structp, png_const_charp) {}

// The header, then (sizes known and checked) the pixels. Only plain data lives across the setjmp: the buffers are
// sized between the two libpng phases, outside any longjmp, and libpng writes into them. Returns an error message,
// empty on success.
std::string read_header(Reader& r, std::FILE* f, png_uint_32& w, png_uint_32& h) {
  g_message[0] = '\0';
  if (setjmp(png_jmpbuf(r.png))) return g_message[0] ? g_message : "libpng error";
  png_init_io(r.png, f);
  png_read_info(r.png, r.info);
  int depth = 0, color = 0, interlace = 0;
  png_get_IHDR(r.png, r.info, &w, &h, &depth, &color, &interlace, nullptr, nullptr);
  const bool gray = color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA;
  if (depth == 16 || (gray && depth != 8)) {
    return "unsupported PNG (colour type " + std::to_string(color) + ", bit depth " + std::to_string(depth) +
           "): only 8-bit images are decoded";
  }
  if (static_cast<uint64_t>(w) * h > kMaxPngPixels) return "image too large (" + std::to_string(w) + " x " +
                                                              std::to_string(h) + " pixels)";
  if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(r.png);
  if (gray) png_set_gray_to_rgb(r.png);
  // The palette expansion turns a tRNS chunk into alpha; PIL's convert("RGB") drops it, as any alpha channel.
  if ((color & PNG_COLOR_MASK_ALPHA) || (color == PNG_COLOR_TYPE_PALETTE && png_get_valid(r.png, r.info, PNG_INFO_tRNS))) {
    png_set_strip_alpha(r.png);
  }
  png_set_interlace_handling(r.png);
  png_read_update_info(r.png, r.info);
  if (png_get_rowbytes(r.png, r.info) != static_cast<size_t>(w) * 3) return "unexpected row size";
  return {};
}

std::string read_rows(Reader& r, std::vector<png_bytep>& rows) {
  g_message[0] = '\0';
  if (setjmp(png_jmpbuf(r.png))) return g_message[0] ? g_message : "libpng error";
  png_read_image(r.png, rows.data());
  png_read_end(r.png, nullptr);
  return {};
}

}  // namespace

Image read_png_rgb(const std::string& path) {
  File file;
  file.f = std::fopen(path.c_str(), "rb");
  if (!file.f) throw AssetError(path + ": cannot open the PNG file");
  unsigned char sig[8];
  if (std::fread(sig, 1, 8, file.f) != 8 || png_sig_cmp(sig, 0, 8) != 0) throw AssetError(path + ": not a PNG file");
  std::rewind(file.f);
  Reader r;
  r.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, on_error, on_warning);
  if (!r.png) throw std::bad_alloc();
  r.info = png_create_info_struct(r.png);
  if (!r.info) throw std::bad_alloc();
  png_uint_32 w = 0, h = 0;
  std::string err = read_header(r, file.f, w, h);
  if (!err.empty()) throw AssetError(path + ": " + err);
  Image out;
  out.height = h;
  out.width = w;
  out.rgb.resize(static_cast<size_t>(w) * h * 3);
  std::vector<png_bytep> rows(h);
  for (png_uint_32 y = 0; y < h; ++y) rows[y] = out.rgb.data() + static_cast<size_t>(y) * w * 3;
  err = read_rows(r, rows);
  if (!err.empty()) throw AssetError(path + ": " + err);
  return out;
}

}  // namespace balla::scene::assets
