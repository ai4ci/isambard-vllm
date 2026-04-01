 
## Storage spaces

## Overview

Each project and user has storage space allocated to them for the duration of the project. Each area is intended for specific data usage, and the performance characteristics and quota limitations of each area reflect that expected usage.

Various storage spaces are provided to projects and its members, and are allocated upon creation of a project in the [BriCS portal](https://portal.isambard.ac.uk/login/) or upon acceptance of an invitation to a project. The underlying physical storage and policies applied to them vary across the BriCS facilities with [Lustre](https://www.lustre.org/), VAST Data (NFS), and local EXT4 being used. The storage areas are organised into two types, Parallel Access Storage, which is share across all nodes within a cluster, and Local Access Storage, which is physically installed into a node, and thus unique to each node. Details of which mounts referenced by their environment variable (i.e. run 'echo $HOME' to see the path) are which are detailed below.

### Parallel Access Storage Areas

The storage areas listed below are presented to all compute and login nodes within a cluster. The files written on one node will be visible on all other nodes.

#### $HOME

User-specific storage area for user's data (e.g. configuration files, submission scripts, job output files). It is owned by the user and, by default, is accessible to the user and members of their project group according to the filesystem permissions, and is not intended for storage of large volumes of data.

#### $SCRATCHDIR

User-specific storage area for working data (e.g. job checkpoint data, input/output data for intermediate processing steps, container images). This is intended as a user's working space for short-lived data that supports running jobs.

Isambard 3 deletes files from $SCRATCHDIR that have not been accessed for 60 days.

#### $PROJECTDIR

Project-specific storage area, for data sharing amongst project members (e.g. input datasets, shared Conda environments, shared container images). The storage is accessible only to members of the project.

#### $PROJECTDIR\_PUBLIC

Project-specific storage area, for "public shared" data, for sharing files with other projects. The directory is readable by *all* users on the system, including members of other projects, but only users of the owning project have write permission.

### Local Access Storage Areas

The storage listed below is accessible only locally on the node that files are written on. Unlike the parallel access storage areas, the disks are physically located in the nodes, and thus each area on a node is separate from every other node's local access storage.

#### $LOCALDIR

Fast local scratch storage on nodes. On compute nodes, `$LOCALDIR` is typically implemented as a tmpfs RAM disk, so available capacity is limited by the node's memory but I/O is very fast; on login nodes, `$LOCALDIR` is backed by local solid-state disks, which offer different performance and capacity characteristics. Temporary storage space intended for use in situations where the shared filesystem is unsuitable (such as [building rootless containers](https://github.com/containers/podman/blob/main/troubleshooting.md#14-rootless-podman-build-fails-eperm-on-nfs)). This is ephemeral storage, which is wiped at the end of a job/session, and its size and implementation may differ between login and compute nodes.

## Paths and Permissions

Below is the path of each storage area, and its associated environment variable (env), default filesystem ownership details, and default file system permissions (expressed in octal notation).

| Path | Env | Default Ownership | Default File Perms |
| --- | --- | --- | --- |
| `/home/<PROJECT>/<USER>.<PROJECT>` | [$HOME](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#home) | Owner: User, Group: Project | Octal: 0750 |
| `/scratch/<PROJECT>/<USER>.<PROJECT>` | [$SCRATCHDIR](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#scratchdir), [$SCRATCH](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#scratchdir) | Owner: User, Group: User | Octal: 0750 |
| `/projects/<PROJECT>` | [$PROJECTDIR](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#projectdir) | Owner: Root, Group: Project | Octal: 2770 |
| `/projects/public/<PROJECT>` | [$PROJECTDIR\_PUBLIC](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#projectdir_public) | Owner: Root, Group: Project | Octal: 2775 |
| `/local/user/<UID>` [^1] | [$LOCALDIR](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#localdir) | Owner: User, Group: User | Octal: 0750 [^2] |

## Quotas

### Capacity Limits

Quotas are enforced across all file systems to ensure runaway usage does not impact other users.

| Storage Space | Isambard 3 | Isambard-AI Phase 1 | Isambard-AI Phase 2 |
| --- | --- | --- | --- |
| $HOME | 100GiB | 100GiB | 100GiB (NFS) [^3] |
| $PROJECTDIR | 20TiB | 200TiB | 200TiB |
| $SCRATCHDIR | 5TiB | 5TiB | 5TiB |
| $LOCALDIR (compute node) | 512GiB | 48GiB | 48GiB |
| $LOCALDIR (login node) | 512GiB | 512GiB | 512GiB |

### File (inode) Limits

In addition to the space limits shown above, file systems also enforce limits on the number of files and directories (inode quotas). Historically these were documented with example values such as `100Mi soft / 105Mi hard`. These inode (file-count) limits remain important for workloads that create very large numbers of small files, even when total space usage is well below the size quota.

The precise inode quotas may vary between systems and over time and are therefore not listed per-filesystem in the table above. If your workflow relies on creating many small files, or if you previously planned around specific inode limits (for example values like `100Mi soft / 105Mi hard`), please consult the current service documentation or contact the support team to confirm the applicable inode quotas for your project.

### Checking Quotas

Lustre quotas are based on Lustre Project Quotas. They can be checked by getting the Project ID of a directory, and then listing the quota information for that ID.

To check a quota you can run the following one line command: `lfs quota -hp $(lfs project -d <DIRECTORY> | awk '{print $1}') <DIRECTORY>`

As an example, to check scratch space: `lfs quota -hp $(lfs project -d $SCRATCHDIR | awk '{print $1}') $SCRATCHDIR`

## Data Retention Policies

All storage space is working storage, thus is not backed up, nor is intended for long-term archival of data.

The data written out to the various storage areas is generally retained until the end date of a project. There are a few notable exceptions to this, which can be viewed in the table hereunder:

| Storage Space | Isambard 3 | Isambard-AI Phase 1 | Isambard-AI Phase 2 |
| --- | --- | --- | --- |
| $HOME | Project End Date | Project End Date | Project End Date |
| $PROJECTDIR | Project End Date | Project End Date | Project End Date |
| $SCRATCHDIR | Last Access >60 days | Project End Date | Project End Date |
| $LOCALDIR (compute node) | End of Job | End of Job | End of Job |
| $LOCALDIR (login node) | End of Session | End of Session | End of Session |

## Important Considerations

Storage expires at the project end date

The storage allocated to each user and project is only accessible for the duration of the project.

After a project's end date project members will no longer able to access or use any storage previously allocated to the project and any data remaining in the project storage area will be deleted. **This applies to project-specific shared storage under `/projects` as well as user storage under `/home/<PROJECT>` and `/scratch/<PROJECT>`.**

All storage is working storage

**Storage on BriCS facilities is working storage. It is not backed up and is not intended for long-term or archival storage of data.**

Please ensure that important data is regularly backed up in another location during the project and that any data that should remain accessible to project members after the end of the project is copied off the system before the project end date.

Isambard 3 Grace and MACS

The filesystem (Lustre) is shared between Isambard 3 Grace and Isambard 3 MACS. The architectures of the system are different (`aarch64` vs. `x86_64`). Any files that may be used on both, such as `.bashrc`, should have appropriate checks, e.g. inside `.bashrc` architecture specific commands, such as Conda initialisation, can be protected by:

```js
if [ "$(arch)" == "x86_64" ]; then
...
elif [ "$(arch)" == "aarch64" ]; then
...
fi
```

This can also include Python `pip` environments installed by `--user` which are installed in `$HOME/.local`. Use of virtual environments should be used instead.

[^1]: UID can be found by running 'echo $UID'

[^2]: LOCALDIR on compute on Isambard-AI has octal 0700.

[^3]: Isambard-AI Phase 2 has its home directories ($HOME) on NFS storage, which unfortunately does not provide a means to display the quota usage. The nearest equivalent is to check the usage against the figures in the [Limits](https://docs.isambard.ac.uk/user-documentation/information/system-storage/#capacity-limits) table, with the following command: `du -xsh $HOME`
