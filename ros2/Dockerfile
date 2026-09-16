FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-c"]
ARG AGX_ROS_COMMIT=b9ad14de2a6eb818a2d206fbd1adba48343662f7
ARG AGX_SDK_COMMIT=e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip python3-colcon-common-extensions python3-yaml \
    ros-humble-moveit ros-humble-moveit-configs-utils \
    ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-topic-based-ros2-control ros-humble-xacro \
    ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir 'numpy<2' 'mujoco==3.2.7' python-dotenv \
    "pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm.git@${AGX_SDK_COMMIT}"
WORKDIR /opt/nero_ws
RUN git clone --branch ros2 https://github.com/agilexrobotics/agx_arm_ros.git src/agx_arm_ros \
    && cd src/agx_arm_ros && git checkout "$AGX_ROS_COMMIT" && git submodule update --init --recursive
RUN source /opt/ros/humble/setup.bash && colcon build \
    --base-paths src/agx_arm_ros/src/agx_arm_description src/agx_arm_ros/src/agx_arm_moveit \
    --packages-select agx_arm_description agx_arm_moveit \
    --cmake-args -DBUILD_TESTING=OFF
COPY entrypoint.sh /nero_entrypoint.sh
RUN chmod +x /nero_entrypoint.sh
ENV ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=73 PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["/nero_entrypoint.sh"]
CMD ["python3", "-m", "nero_agent", "doctor"]
