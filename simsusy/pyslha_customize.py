import pyslha
from typing import Dict, Optional, Sequence, List, Tuple, Union, Any, MutableMapping  # noqa: F401

KeyType = Union[None, int, Tuple[int, int]]
ValueType = Union[int, float, str, List[str]]   # SPINFO/DCINFO 3 and 4 may be multiple
CommentType = str

BLOCKS_ORDER = [
    'SPINFO', 'DCINFO', 'MODSEL', 'SMINPUTS', 'MINPAR', 'EXTPAR',
    'VCKMIN', 'UPMNSIN', 'MSQ2IN', 'MSU2IN', 'MSD2IN', 'MSL2IN', 'MSE2IN', 'TUIN', 'TDIN', 'TEIN',
    'MASS', 'NMIX', 'UMIX', 'VMIX', 'ALPHA', 'HMIX', 'GAUGE', 'MSOFT',
    'MSQ2', 'MSU2', 'MSD2', 'MSL2', 'MSE2',
    'STOPMIX', 'SBOTMIX', 'STAUMIX', 'USQMIX', 'DSQMIX', 'SELMIX', 'SNUMIX',
    'AU', 'AD', 'AE', 'TU', 'TD', 'TE', 'YU', 'YD', 'YE',
]


def _comment_str(comment: CommentType='')->str:
    comment = comment.strip()
    return '# ' + comment + '\n' if comment else '#\n'


def format_block_line(name: str, q: Optional[float]=None, comment: CommentType='')->str:
    q_str = '' if q is None else f'Q={q:15.8e}'
    line = f'BLOCK {name.upper()} {q_str}'
    return f'{line:25}   {_comment_str(comment)}'


def format_line(key: KeyType, value: ValueType, comment: CommentType='')->str:
    if isinstance(value, list):  # for SPINFO / DCINFO 3 and 4
        return '\n'.join([format_line(key, line) for line in value])

    if isinstance(value, float):
        value_str = f'{value:16.8e}'
    elif isinstance(value, int):
        value_str = f'{value:>10}      '
    else:
        value_str = value
    if isinstance(key, int):
        # (1x,I5,3x,1P,E16.8,0P,3x,'#',1x,A)
        return f' {key:>5}   {value_str}   {_comment_str(comment)}'
    elif isinstance(key, tuple):
        # (1x,I2,1x,I2,3x,1P,E16.8,0P,3x,'#',1x,A)
        return f' {key[0]:>2} {key[1]:>2}   {value_str}   {_comment_str(comment)}'
    else:
        # (9x,1P,E16.8,0P,3x,'#',1x,A)
        return f'         {value_str}   {_comment_str(comment)}'


def format_mass_line(key: KeyType, value: ValueType, comment: CommentType='')->str:
    return f' {key:>9}   {value:15.8e}   {_comment_str(comment)}'


def writeSLHABlocks(blocks: MutableMapping[Tuple[str, ...], pyslha.Block], precision: int=8) -> str:
    order = list(BLOCKS_ORDER)  # clone
    for b in blocks.values():
        if b.name not in order:
            order.append(b.name)
    lines = []
    for name in order:
        b = blocks.get(tuple(name))
        if b:
            lines.append(format_block_line(b.name, b.q))
            if b.name.upper() == 'MASS':
                lines += [format_mass_line(k, b[k]) for k in sorted(b.keys())]
            else:
                lines += [format_line(k, b[k]) for k in sorted(b.keys())]
            lines.append('\n')
    return ''.join(lines)


def writeSLHADecays(decays: MutableMapping[int, pyslha.Particle], ignorenobr: bool=False, precision: int=8)->str:
    lines = []
    if decays is None:
        return ''
    for pid, particle in decays.items():
        lines.append(f'DECAY {particle.pid:>9}   {particle.totalwidth:16.8e}   #\n')
        for d in particle.decays:
            if d.br > 0 or not ignorenobr:
                ids_str = ''.join([f'{i:>9} ' for i in d.ids])
                lines.append(f'   {d.br:16.8e}   {len(d.ids):>2}   {ids_str}  #\n')
        lines.append('\n')
    return ''.join(lines)
