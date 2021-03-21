#!/usr/bin/env bash
#
# For multi-agent policies: ./download-pretrained.sh
# For single-agent policies: ./download-pretrained.sh --single-agent

if [[ $# -eq 0 || "$1" != "--single-agent" ]]; then
    # Multi-agent policies (16 total)
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092757904081-lifting_4-small_empty-ours/config.yml -P logs/20201214T092757904081-lifting_4-small_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml -P logs/20201217T171233203789-lifting_4-small_divider-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092812731965-lifting_4-large_empty-ours/config.yml -P logs/20201214T092812731965-lifting_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092812756074-lifting_4-large_doors-ours/config.yml -P logs/20201214T092812756074-lifting_4-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092757944372-lifting_4-large_tunnels-ours/config.yml -P logs/20201214T092757944372-lifting_4-large_tunnels-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092812688379-lifting_4-large_rooms-ours/config.yml -P logs/20201214T092812688379-lifting_4-large_rooms-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092757850335-pushing_4-small_empty-ours/config.yml -P logs/20201214T092757850335-pushing_4-small_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092814688334-pushing_4-small_divider-ours/config.yml -P logs/20201214T092814688334-pushing_4-small_divider-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171253620771-pushing_4-large_empty-ours/config.yml -P logs/20201217T171253620771-pushing_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092812868257-lifting_2_pushing_2-large_empty-ours/config.yml -P logs/20201214T092812868257-lifting_2_pushing_2-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171253875360-lifting_2_pushing_2-large_doors-ours/config.yml -P logs/20201217T171253875360-lifting_2_pushing_2-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171240497725-lifting_2_pushing_2-large_rooms-ours/config.yml -P logs/20201217T171240497725-lifting_2_pushing_2-large_rooms-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171253796927-lifting_2_throwing_2-large_empty-ours/config.yml -P logs/20201217T171253796927-lifting_2_throwing_2-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171239448464-lifting_2_throwing_2-large_doors-ours/config.yml -P logs/20201217T171239448464-lifting_2_throwing_2-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20210119T200133911206-rescue_4-large_empty-ours/config.yml -P logs/20210119T200133911206-rescue_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20210120T031916058932-rescue_4-small_empty-ours/config.yml -P logs/20210120T031916058932-rescue_4-small_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092757904081-lifting_4-small_empty-ours/policy_00164000.pth.tar -P checkpoints/20201214T092757904081-lifting_4-small_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171233203789-lifting_4-small_divider-ours/policy_00164000.pth.tar -P checkpoints/20201217T171233203789-lifting_4-small_divider-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092812731965-lifting_4-large_empty-ours/policy_00164000.pth.tar -P checkpoints/20201214T092812731965-lifting_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092812756074-lifting_4-large_doors-ours/policy_00164000.pth.tar -P checkpoints/20201214T092812756074-lifting_4-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092757944372-lifting_4-large_tunnels-ours/policy_00164000.pth.tar -P checkpoints/20201214T092757944372-lifting_4-large_tunnels-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092812688379-lifting_4-large_rooms-ours/policy_00164000.pth.tar -P checkpoints/20201214T092812688379-lifting_4-large_rooms-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092757850335-pushing_4-small_empty-ours/policy_00246000.pth.tar -P checkpoints/20201214T092757850335-pushing_4-small_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092814688334-pushing_4-small_divider-ours/policy_00246000.pth.tar -P checkpoints/20201214T092814688334-pushing_4-small_divider-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171253620771-pushing_4-large_empty-ours/policy_00246000.pth.tar -P checkpoints/20201217T171253620771-pushing_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092812868257-lifting_2_pushing_2-large_empty-ours/policy_00246000.pth.tar -P checkpoints/20201214T092812868257-lifting_2_pushing_2-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171253875360-lifting_2_pushing_2-large_doors-ours/policy_00246000.pth.tar -P checkpoints/20201217T171253875360-lifting_2_pushing_2-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171240497725-lifting_2_pushing_2-large_rooms-ours/policy_00246000.pth.tar -P checkpoints/20201217T171240497725-lifting_2_pushing_2-large_rooms-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171253796927-lifting_2_throwing_2-large_empty-ours/policy_00164000.pth.tar -P checkpoints/20201217T171253796927-lifting_2_throwing_2-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171239448464-lifting_2_throwing_2-large_doors-ours/policy_00164000.pth.tar -P checkpoints/20201217T171239448464-lifting_2_throwing_2-large_doors-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20210119T200133911206-rescue_4-large_empty-ours/policy_00015375.pth.tar -P checkpoints/20210119T200133911206-rescue_4-large_empty-ours
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20210120T031916058932-rescue_4-small_empty-ours/policy_00015375.pth.tar -P checkpoints/20210120T031916058932-rescue_4-small_empty-ours
else
    # Single-agent policies (11 total)
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171254022070-lifting_1-small_empty-base/config.yml -P logs/20201217T171254022070-lifting_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171253990084-lifting_1-small_divider-base/config.yml -P logs/20201217T171253990084-lifting_1-small_divider-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201217T171253912482-lifting_1-large_empty-base/config.yml -P logs/20201217T171253912482-lifting_1-large_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092758744624-lifting_1-large_doors-base/config.yml -P logs/20201214T092758744624-lifting_1-large_doors-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092813523160-lifting_1-large_tunnels-base/config.yml -P logs/20201214T092813523160-lifting_1-large_tunnels-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092815711154-lifting_1-large_rooms-base/config.yml -P logs/20201214T092815711154-lifting_1-large_rooms-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092813073846-pushing_1-small_empty-base/config.yml -P logs/20201214T092813073846-pushing_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092815287579-pushing_1-small_divider-base/config.yml -P logs/20201214T092815287579-pushing_1-small_divider-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20201214T092813153074-pushing_1-large_empty-base/config.yml -P logs/20201214T092813153074-pushing_1-large_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20210119T200131797089-rescue_1-small_empty-base/config.yml -P logs/20210119T200131797089-rescue_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/logs/20210120T031934370292-rescue_1-large_empty-base/config.yml -P logs/20210120T031934370292-rescue_1-large_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171254022070-lifting_1-small_empty-base/policy_00041000.pth.tar -P checkpoints/20201217T171254022070-lifting_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171253990084-lifting_1-small_divider-base/policy_00041000.pth.tar -P checkpoints/20201217T171253990084-lifting_1-small_divider-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201217T171253912482-lifting_1-large_empty-base/policy_00041000.pth.tar -P checkpoints/20201217T171253912482-lifting_1-large_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092758744624-lifting_1-large_doors-base/policy_00041000.pth.tar -P checkpoints/20201214T092758744624-lifting_1-large_doors-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092813523160-lifting_1-large_tunnels-base/policy_00041000.pth.tar -P checkpoints/20201214T092813523160-lifting_1-large_tunnels-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092815711154-lifting_1-large_rooms-base/policy_00041000.pth.tar -P checkpoints/20201214T092815711154-lifting_1-large_rooms-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092813073846-pushing_1-small_empty-base/policy_00061500.pth.tar -P checkpoints/20201214T092813073846-pushing_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092815287579-pushing_1-small_divider-base/policy_00061500.pth.tar -P checkpoints/20201214T092815287579-pushing_1-small_divider-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20201214T092813153074-pushing_1-large_empty-base/policy_00061500.pth.tar -P checkpoints/20201214T092813153074-pushing_1-large_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20210119T200131797089-rescue_1-small_empty-base/policy_00003844.pth.tar -P checkpoints/20210119T200131797089-rescue_1-small_empty-base
    wget -c https://spatial-intention-maps.cs.princeton.edu/pretrained/checkpoints/20210120T031934370292-rescue_1-large_empty-base/policy_00003844.pth.tar -P checkpoints/20210120T031934370292-rescue_1-large_empty-base
fi
