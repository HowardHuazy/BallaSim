// The ROS message definitions the exporters write (docs/design/phase3/outputs.md §4.8). For ROS1 (rosbag1, A6), each
// type's name, MD5 and message_definition: the full text of genmsg's compute_full_text (gentools.py at f64190b), i.e.
// the type's own .msg text, then for each dependency in depth-first first-use order a line of 80 '=', "MSG: <type>"
// and that type's text, with the last newline dropped. Each type carries its own dependency texts. The MD5s follow
// genmsg's rule (R-FMT out/t1_ros1_md5.json, which reproduced 10 of 10 recorded EVIMO2 connection headers);
// tests/python/test_export.py (F2) re-derives them from these texts. For ROS 2 (MCAP, A7), each type's name and its
// ros2msg schema (the MCAP ros2 profile): the type's .msg fields, then for each dependency in depth-first first-use
// order a line of 80 '=', "MSG: <pkg>/<Type>" and that type's fields (§4.8), with the last newline dropped. The ROS 2
// texts are the field declarations only (types, names, defaults), without the upstream comments: a ros2msg reader
// takes the fields from them, ROS 2 tools take the type from the schema name, and the RIHS01 type hash that could
// depend on them is left out (outputs.md Q11). A9 adds sensor_msgs/Image.
// Text sources (no ROS code is used; the texts are interface definitions, listed with their licences in
// THIRD_PARTY_NOTICES by P3-A A10):
//   sensor_msgs/Imu, std_msgs/Header, geometry_msgs/Quaternion, geometry_msgs/Vector3: the full text ROS recorded in
//     EVIMO2's bags (connection /prophesee/left/imu of imo/train/scene13_dyn_test_01; ros/common_msgs, BSD-3-Clause),
//     verbatim, trailing spaces included;
//   dvs_msgs/Event, dvs_msgs/EventArray: uzh-rpg/rpg_dvs_ros (MIT) as R-FMT scripts/ros1.py quotes them; they equal,
//     but for the package, the prophesee_event_msgs texts recorded in the same bag;
//   event_camera_msgs/EventPacket: ros-event-camera/event_camera_msgs (Apache-2.0) at 24cc8da as R-FMT scripts/ros1.py
//     quotes it (not re-read for this increment);
//   sensor_msgs/Image (A9): the field lines of ros/common_msgs' Image.msg (BSD-3-Clause) as R-FMT scripts/ros1.py
//     quotes them, the upstream comments left out (no copy of the .msg is on this machine to quote; comments change
//     neither genmsg's MD5, which R-FMT out/t1_ros1_md5.json derives as 060021388200f6f0f447d0fcd9c64743, nor the
//     layout); its ROS 2 fields, sensor_msgs/msg/Image (ros2/common_interfaces, Apache-2.0), likewise;
//   ROS 2 (A7): the fields of sensor_msgs/msg/Imu, std_msgs/msg/Header, builtin_interfaces/msg/Time,
//     geometry_msgs/msg/{Quaternion, Vector3} (ros2/common_interfaces, rcl_interfaces; Apache-2.0) and
//     event_camera_msgs/msg/EventPacket, whose CDR layout R-FMT scripts/mcap2.py serializes (its golden CDR Imu
//     carries the ROS 2 Quaternion's default w = 1); written from those layouts, not re-read upstream (no network).
// Header-only.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "export/errors.h"

namespace balla::exporter {

// The message types of the ROS writers (§3.3); kImage (sensor_msgs/Image) since P3-A A9.
enum class MsgType : uint8_t { kDvsEventArray, kEventPacketEvt3, kImu, kImage };

// The sensor_msgs/Imu covariances (§4.8): kModel, the diagonal per-sample white-noise variances; kZero, all zero
// ("covariance unknown", as EVIMO2 recorded them).
enum class CovarianceMode : uint8_t { kModel, kZero };

namespace msgdefs {

inline constexpr std::string_view kHeader =  // std_msgs/Header
    "# Standard metadata for higher-level stamped data types.\n"
    "# This is generally used to communicate timestamped data \n"
    "# in a particular coordinate frame.\n"
    "# \n"
    "# sequence ID: consecutively increasing ID \n"
    "uint32 seq\n"
    "#Two-integer timestamp that is expressed as:\n"
    "# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
    "# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
    "# time-handling sugar is provided by the client library\n"
    "time stamp\n"
    "#Frame this data is associated with\n"
    "string frame_id\n";

inline constexpr std::string_view kQuaternion =  // geometry_msgs/Quaternion
    "# This represents an orientation in free space in quaternion form.\n"
    "\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n"
    "float64 w\n";

inline constexpr std::string_view kVector3 =  // geometry_msgs/Vector3
    "# This represents a vector in free space. \n"
    "# It is only meant to represent a direction. Therefore, it does not\n"
    "# make sense to apply a translation to it (e.g., when applying a \n"
    "# generic rigid transformation to a Vector3, tf2 will only apply the\n"
    "# rotation). If you want your data to be translatable too, use the\n"
    "# geometry_msgs/Point message instead.\n"
    "\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

inline constexpr std::string_view kImu =  // sensor_msgs/Imu
    "# This is a message to hold data from an IMU (Inertial Measurement Unit)\n"
    "#\n"
    "# Accelerations should be in m/s^2 (not in g's), and rotational velocity should be in rad/sec\n"
    "#\n"
    "# If the covariance of the measurement is known, it should be filled in (if all you know is the \n"
    "# variance of each measurement, e.g. from the datasheet, just put those along the diagonal)\n"
    "# A covariance matrix of all zeros will be interpreted as \"covariance unknown\", and to use the\n"
    "# data a covariance will have to be assumed or gotten from some other source\n"
    "#\n"
    "# If you have no estimate for one of the data elements (e.g. your IMU doesn't produce an orientation \n"
    "# estimate), please set element 0 of the associated covariance matrix to -1\n"
    "# If you are interpreting this message, please check for a value of -1 in the first element of each \n"
    "# covariance matrix, and disregard the associated estimate.\n"
    "\n"
    "Header header\n"
    "\n"
    "geometry_msgs/Quaternion orientation\n"
    "float64[9] orientation_covariance # Row major about x, y, z axes\n"
    "\n"
    "geometry_msgs/Vector3 angular_velocity\n"
    "float64[9] angular_velocity_covariance # Row major about x, y, z axes\n"
    "\n"
    "geometry_msgs/Vector3 linear_acceleration\n"
    "float64[9] linear_acceleration_covariance # Row major x, y z \n";

inline constexpr std::string_view kDvsEvent =  // dvs_msgs/Event
    "# A DVS event\n"
    "uint16 x\n"
    "uint16 y\n"
    "time ts\n"
    "bool polarity\n";

inline constexpr std::string_view kDvsEventArray =  // dvs_msgs/EventArray
    "# This message contains an array of events\n"
    "# (0, 0) is at top-left corner of image\n"
    "#\n"
    "\n"
    "Header header\n"
    "\n"
    "uint32 height         # image height, that is, number of rows\n"
    "uint32 width          # image width, that is, number of columns\n"
    "\n"
    "# an array of events\n"
    "Event[] events\n";

inline constexpr std::string_view kEventPacket =  // event_camera_msgs/EventPacket
    "# This message contains a packet of camera events\n"
    "\n"
    "Header header     # standard ROS header\n"
    "uint32 height     # image height\n"
    "uint32 width      # image width\n"
    "\n"
    "uint64 seq        # sequence number to detect lost messages\n"
    "uint64 time_base  # event ros time = time_base + decoded event_time\n"
    "string encoding   # the encoding scheme to use\n"
    "bool is_bigendian # true if machine stores in big endian format\n"
    "\n"
    "uint8[] events\n";

inline constexpr std::string_view kImage =  // sensor_msgs/Image (field lines)
    "Header header\n"
    "uint32 height\n"
    "uint32 width\n"
    "string encoding\n"
    "uint8 is_bigendian\n"
    "uint32 step\n"
    "uint8[] data\n";

inline constexpr std::string_view kSeparator =
    "================================================================================\n";

// genmsg's full text of a type from its own text and its dependencies' (name, text), in depth-first first-use order.
inline std::string full_text(std::string_view own,
                             std::initializer_list<std::pair<std::string_view, std::string_view>> deps) {
  std::string s(own);
  s += '\n';
  for (const auto& [name, text] : deps) {
    s += kSeparator;
    s += "MSG: ";
    s += name;
    s += '\n';
    s += text;
    s += '\n';
  }
  s.pop_back();
  return s;
}

// ROS 2 field declarations (ros2msg).
inline constexpr std::string_view kTime2 =  // builtin_interfaces/msg/Time
    "int32 sec\n"
    "uint32 nanosec\n";

inline constexpr std::string_view kHeader2 =  // std_msgs/msg/Header
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n";

inline constexpr std::string_view kQuaternion2 =  // geometry_msgs/msg/Quaternion
    "float64 x 0\n"
    "float64 y 0\n"
    "float64 z 0\n"
    "float64 w 1\n";

inline constexpr std::string_view kVector32 =  // geometry_msgs/msg/Vector3
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

inline constexpr std::string_view kImu2 =  // sensor_msgs/msg/Imu
    "std_msgs/Header header\n"
    "geometry_msgs/Quaternion orientation\n"
    "float64[9] orientation_covariance\n"
    "geometry_msgs/Vector3 angular_velocity\n"
    "float64[9] angular_velocity_covariance\n"
    "geometry_msgs/Vector3 linear_acceleration\n"
    "float64[9] linear_acceleration_covariance\n";

inline constexpr std::string_view kEventPacket2 =  // event_camera_msgs/msg/EventPacket
    "std_msgs/Header header\n"
    "uint32 height\n"
    "uint32 width\n"
    "uint64 seq\n"
    "uint64 time_base\n"
    "string encoding\n"
    "bool is_bigendian\n"
    "uint8[] events\n";

inline constexpr std::string_view kImage2 =  // sensor_msgs/msg/Image
    "std_msgs/Header header\n"
    "uint32 height\n"
    "uint32 width\n"
    "string encoding\n"
    "uint8 is_bigendian\n"
    "uint32 step\n"
    "uint8[] data\n";

}  // namespace msgdefs

// One ROS1 message type: its name, its MD5 (genmsg) and its message_definition (the full text).
struct Ros1MsgDef {
  std::string_view type, md5;
  std::string_view definition;
};

inline const Ros1MsgDef& ros1_msgdef(MsgType t) {
  using namespace msgdefs;
  static const std::string dvs =
      full_text(kDvsEventArray, {{"std_msgs/Header", kHeader}, {"dvs_msgs/Event", kDvsEvent}});
  static const std::string packet = full_text(kEventPacket, {{"std_msgs/Header", kHeader}});
  static const std::string imu =
      full_text(kImu, {{"std_msgs/Header", kHeader},
                       {"geometry_msgs/Quaternion", kQuaternion},
                       {"geometry_msgs/Vector3", kVector3}});
  static const std::string image = full_text(kImage, {{"std_msgs/Header", kHeader}});
  static const Ros1MsgDef defs[4] = {{"dvs_msgs/EventArray", "5e8beee5a6c107e504c2e78903c224b8", dvs},
                                     {"event_camera_msgs/EventPacket", "8a6101b351a64529ac77bd8b71074e52", packet},
                                     {"sensor_msgs/Imu", "6a62c6daae103f4ff57a132d6f95cec2", imu},
                                     {"sensor_msgs/Image", "060021388200f6f0f447d0fcd9c64743", image}};
  switch (t) {
    case MsgType::kDvsEventArray: return defs[0];
    case MsgType::kEventPacketEvt3: return defs[1];
    case MsgType::kImu: return defs[2];
    case MsgType::kImage: return defs[3];
  }
  throw Refused("unknown ROS1 message type");
}

// One ROS 2 message type as an MCAP ros2 schema: its name (pkg/msg/Type) and its ros2msg text.
struct Ros2MsgDef {
  std::string_view type;
  std::string_view definition;
};

// The ROS 2 types the MCAP writer stores; there is no ROS 2 dvs_msgs (outputs.md §1.3), so kDvsEventArray is refused.
inline const Ros2MsgDef& ros2_msgdef(MsgType t) {
  using namespace msgdefs;
  static const std::string packet =
      full_text(kEventPacket2, {{"std_msgs/Header", kHeader2}, {"builtin_interfaces/Time", kTime2}});
  static const std::string imu = full_text(kImu2, {{"std_msgs/Header", kHeader2},
                                                   {"builtin_interfaces/Time", kTime2},
                                                   {"geometry_msgs/Quaternion", kQuaternion2},
                                                   {"geometry_msgs/Vector3", kVector32}});
  static const std::string image =
      full_text(kImage2, {{"std_msgs/Header", kHeader2}, {"builtin_interfaces/Time", kTime2}});
  static const Ros2MsgDef defs[3] = {{"event_camera_msgs/msg/EventPacket", packet},
                                     {"sensor_msgs/msg/Imu", imu},
                                     {"sensor_msgs/msg/Image", image}};
  switch (t) {
    case MsgType::kEventPacketEvt3: return defs[0];
    case MsgType::kImu: return defs[1];
    case MsgType::kImage: return defs[2];
    case MsgType::kDvsEventArray: throw Refused("dvs_msgs/EventArray has no ROS 2 type: use EventPacket evt3");
  }
  throw Refused("unknown ROS 2 message type");
}

}  // namespace balla::exporter
