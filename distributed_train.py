#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import socket
import subprocess
import json
from train import distributed_main as single_process_main
from fairseq import distributed_utils, options
import random

def main(args):
    node_to_rank = json.load(open('node_to_rank.json', 'r'))
    args.master_addr = {v:k for k,v in node_to_rank.items()}[0] 
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '10000'
    host = socket.gethostbyname(socket.gethostname())
    args.distributed_port = '10000'
    print("master", args.master_addr, "host", host, os.environ.get('SLURM_JOB_NODELIST'))
    print("OMPI_COMM_WORLD_SIZE", os.environ["OMPI_COMM_WORLD_SIZE"])
    print("OMPI_COMM_WORLD_RANK", os.environ["OMPI_COMM_WORLD_RANK"])
    print("OMPI_COMM_WORLD_LOCAL_RANK", os.environ["OMPI_COMM_WORLD_LOCAL_RANK"], args.device_id)
    exp_id = args.master_addr 
    args.distributed_init_method = "file:///shared/share/" + (exp_id)
    print('| initialized host {} as rank {}'.format(socket.gethostbyname(socket.gethostname()), args.distributed_rank))
    single_process_main(0, args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
