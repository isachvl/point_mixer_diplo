#!/usr/bin/env python3
"""Run PointMixer semantic inference on one ScanNet++ scene."""

import argparse
import csv
import colorsys
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.utils.voxelize import voxelize
from model import get as get_model


PLY_TYPES = {
    'char': 'i1',
    'int8': 'i1',
    'uchar': 'u1',
    'uint8': 'u1',
    'short': '<i2',
    'int16': '<i2',
    'ushort': '<u2',
    'uint16': '<u2',
    'int': '<i4',
    'int32': '<i4',
    'uint': '<u4',
    'uint32': '<u4',
    'float': '<f4',
    'float32': '<f4',
    'double': '<f8',
    'float64': '<f8',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-ply', required=True,
                        help='Path to mesh_aligned_0.05.ply or mesh_aligned_0.05_semantic.ply')
    parser.add_argument('--checkpoint', required=True,
                        help='PointMixer Lightning .ckpt produced by training')
    parser.add_argument('--label-map', required=True,
                        help='TSV from pointmixer_scannetpp_top100/meta/label_mapping.tsv')
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--classes', type=int, default=0)
    parser.add_argument('--max-points', type=int, default=16000)
    parser.add_argument('--min-points', type=int, default=1024)
    parser.add_argument('--block-size', type=float, default=2.0)
    parser.add_argument('--votes', type=int, default=1)
    parser.add_argument('--confidence-threshold', type=float, default=0.0)
    parser.add_argument('--write-point-csv', action='store_true')
    parser.add_argument('--ignore-label', type=int, default=255)

    parser.add_argument('--nsample', nargs='+', type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--input-voxel-size', type=float, default=0.0,
                        help='Optional deterministic input downsample. Use 0.05 for dense pc_aligned.ply.')
    return parser.parse_args()


def read_ply_header(stream):
    lines = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('PLY header ended before end_header.')
        line = line.decode('ascii').rstrip('\r\n')
        lines.append(line)
        if line == 'end_header':
            return lines


def parse_ply_header(lines):
    ply_format = None
    vertex_count = None
    vertex_props = []
    current_element = None

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            ply_format = parts[1]
        elif parts[0] == 'element':
            current_element = parts[1]
            if current_element == 'vertex':
                vertex_count = int(parts[2])
        elif parts[0] == 'property' and current_element == 'vertex':
            if parts[1] == 'list':
                raise ValueError('List properties in vertex elements are not supported.')
            vertex_props.append((parts[2], parts[1]))

    if ply_format != 'binary_little_endian':
        raise ValueError('Only binary_little_endian PLY is supported, got {}'.format(ply_format))
    if vertex_count is None:
        raise ValueError('PLY has no vertex element.')
    return vertex_count, vertex_props


def read_scene_ply(path):
    with open(path, 'rb') as f:
        header = read_ply_header(f)
        vertex_count, vertex_props = parse_ply_header(header)
        dtype = []
        for name, type_name in vertex_props:
            if type_name not in PLY_TYPES:
                raise ValueError('Unsupported PLY property type: {}'.format(type_name))
            dtype.append((name, PLY_TYPES[type_name]))
        vertices = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)

    required = ['x', 'y', 'z']
    missing = [name for name in required if name not in vertices.dtype.names]
    if missing:
        raise ValueError('{} misses required vertex fields: {}'.format(path, missing))

    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
    if all(name in vertices.dtype.names for name in ['red', 'green', 'blue']):
        rgb = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    return xyz, rgb


def downsample_input(xyz, rgb, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return xyz, rgb, np.arange(xyz.shape[0], dtype=np.int64)

    coord = xyz.astype(np.float32).copy()
    coord -= coord.min(axis=0, keepdims=True)
    idx_sort, count = voxelize(coord, voxel_size, mode=1)
    starts = np.cumsum(np.insert(count, 0, 0)[:-1])
    keep_idx = np.sort(idx_sort[starts]).astype(np.int64)
    return xyz[keep_idx], rgb[keep_idx], keep_idx


def write_prediction_ply(path, xyz, rgb, labels, confidence, source_indices=None):
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', '<i4'), ('confidence', '<f4'), ('source_index', '<i4'),
    ])
    out = np.empty(xyz.shape[0], dtype=dtype)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out['red'], out['green'], out['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    out['label'] = labels.astype(np.int32)
    out['confidence'] = confidence.astype(np.float32)
    if source_indices is None:
        source_indices = np.arange(xyz.shape[0], dtype=np.int32)
    out['source_index'] = source_indices.astype(np.int32)

    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        'element vertex {}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
        'property int label\n'
        'property float confidence\n'
        'property int source_index\n'
        'end_header\n'
    ).format(xyz.shape[0])
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        out.tofile(f)


def read_label_map(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            idx = int(row['class_index'])
            name = row.get('source_label', '') or row.get('class_name', '')
            mapping[idx] = name
    return mapping


def make_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        hue = (idx * 0.61803398875) % 1.0
        sat = 0.72
        val = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[idx] = np.asarray([r * 255, g * 255, b * 255], dtype=np.uint8)
    return palette


def build_model_args(args, classes):
    return SimpleNamespace(
        model='net_pointmixer',
        arch='pointmixer',
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transdown='SymmetricTransitionDownBlock',
        transup='SymmetricTransitionUpBlock',
        nsample=args.nsample,
        downsample=[1, 4, 4, 4, 4],
        drop_rate=0.1,
        fea_dim=6,
        classes=classes,
        ignore_label=args.ignore_label,
        print_freq=1000,
        optim='SGD',
        train_batch=1,
        val_batch=1,
        MYCHECKPOINT=str(args.output_dir),
        voxel_size=args.voxel_size,
        off_text_logger=True,
        on_train=False,
    )


def infer_classes_from_checkpoint(state_dict):
    for key, value in state_dict.items():
        if key.endswith('cls.3.weight'):
            return int(value.shape[0])
    return 0


def is_panoptic_checkpoint(state_dict):
    return any(key.startswith('model.offset.') for key in state_dict)


def load_pointmixer(checkpoint, model_args):
    ckpt = torch.load(checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    panoptic_checkpoint = is_panoptic_checkpoint(state_dict)
    if panoptic_checkpoint:
        state_dict = dict(state_dict)
        state_dict.pop('semantic_class_weight', None)
        model_args.model = 'net_pointmixer_panoptic'
        model_args.arch = 'pointmixer_panoptic'
        model_args.offset_loss_weight = 0.0
        model_args.offset_dir_loss_weight = 0.0
        model_args.train_offset_only = False
        model_args.semantic_loss_type = 'ce'
        model_args.focal_gamma = 2.0
        model_args.focal_loss_weight = 1.0
        model_args.lovasz_loss_weight = 0.0
        model_args.semantic_label_smoothing = 0.0
        model_args.class_weight_path = None
        print('[PM INFER] detected panoptic checkpoint; using semantic_logits only', flush=True)
    model = get_model(model_args.model)(args=model_args)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    return model.model


def split_large_indices(xyz, idx, max_points):
    if idx.shape[0] <= max_points:
        return [idx]
    pts = xyz[idx]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    order = idx[np.argsort(pts[:, axis])]
    return [order[start:start + max_points] for start in range(0, order.shape[0], max_points)]


def build_spatial_chunks(xyz, block_size, max_points, vote, votes):
    xy = xyz[:, :2]
    xy_min = xy.min(axis=0)
    shift = np.asarray([
        ((vote * 0.61803398875) % 1.0) * block_size,
        ((vote * 0.41421356237) % 1.0) * block_size,
    ], dtype=np.float32)
    grid = np.floor((xy - xy_min + shift) / block_size).astype(np.int64)
    order = np.lexsort((grid[:, 1], grid[:, 0]))
    sorted_grid = grid[order]

    change = np.empty(order.shape[0], dtype=bool)
    change[0] = True
    change[1:] = np.any(sorted_grid[1:] != sorted_grid[:-1], axis=1)
    starts = np.where(change)[0]
    ends = np.concatenate([starts[1:], np.asarray([order.shape[0]], dtype=np.int64)])

    chunks = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        chunks.extend(split_large_indices(xyz, order[start:end], max_points))
    return chunks


def prepare_chunk(xyz, feat, idx, min_points):
    actual_n = idx.shape[0]
    if actual_n < min_points:
        pad = np.random.choice(idx, size=min_points - actual_n, replace=True)
        proc_idx = np.concatenate([idx, pad])
    else:
        proc_idx = idx

    coord = xyz[proc_idx].astype(np.float32).copy()
    coord -= coord.min(axis=0, keepdims=True)
    color = feat[proc_idx].astype(np.float32)
    return coord, color, actual_n


@torch.no_grad()
def predict_scene(model, xyz, feat, classes, max_points, min_points, block_size, votes):
    score_sum = np.zeros((xyz.shape[0], classes), dtype=np.float32)
    vote_count = np.zeros((xyz.shape[0],), dtype=np.uint16)

    for vote in range(votes):
        chunks = build_spatial_chunks(xyz, block_size, max_points, vote, votes)
        print('[PM INFER] vote {}/{}: {} chunks'.format(vote + 1, votes, len(chunks)), flush=True)
        for chunk_id, idx in enumerate(chunks, start=1):
            coord, color, actual_n = prepare_chunk(xyz, feat, idx, min_points)
            coord_t = torch.from_numpy(coord).float().cuda(non_blocking=True).contiguous()
            feat_t = torch.from_numpy(color).float().cuda(non_blocking=True).contiguous()
            offset_t = torch.IntTensor([coord.shape[0]]).cuda(non_blocking=True)

            outputs = model([coord_t, feat_t, offset_t])
            logits = outputs['semantic_logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits[:actual_n], dim=1).detach().cpu().numpy()

            score_sum[idx[:actual_n]] += probs
            vote_count[idx[:actual_n]] += 1

            if chunk_id % 25 == 0 or chunk_id == len(chunks):
                print('[PM INFER] chunk {}/{}'.format(chunk_id, len(chunks)), flush=True)

    covered = vote_count > 0
    score_sum[covered] /= vote_count[covered, None].astype(np.float32)
    pred = np.argmax(score_sum, axis=1).astype(np.int32)
    conf = np.max(score_sum, axis=1).astype(np.float32)
    pred[~covered] = 255
    conf[~covered] = 0.0
    return pred, conf


def write_point_csv(path, xyz, labels, names, confidence, colors, kept_idx, source_indices=None):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'point_index', 'source_point_index', 'x', 'y', 'z',
            'class_index', 'class_name', 'confidence',
            'red', 'green', 'blue',
        ])
        for idx in kept_idx.tolist():
            label = int(labels[idx])
            source_idx = int(source_indices[idx]) if source_indices is not None else idx
            writer.writerow([
                idx, source_idx,
                float(xyz[idx, 0]), float(xyz[idx, 1]), float(xyz[idx, 2]),
                label,
                names.get(label, 'class_{}'.format(label)),
                float(confidence[idx]),
                int(colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2]),
            ])


def write_summary_csv(path, labels, names, confidence, kept_idx):
    kept_labels = labels[kept_idx]
    unique, counts = np.unique(kept_labels, return_counts=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_index', 'class_name', 'count', 'mean_confidence'])
        for label, count in zip(unique.tolist(), counts.tolist()):
            mask = kept_labels == label
            conf_mean = float(confidence[kept_idx][mask].mean()) if count > 0 else 0.0
            writer.writerow([label, names.get(label, 'class_{}'.format(label)), count, conf_mean])


def color_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def color_rgb_text(rgb):
    return '{},{},{}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def excel_col_name(col_idx):
    name = ''
    col_idx += 1
    while col_idx:
        col_idx, rem = divmod(col_idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def xml_cell(row_idx, col_idx, value, style_id=0):
    ref = '{}{}'.format(excel_col_name(col_idx), row_idx)
    style = ' s="{}"'.format(style_id) if style_id else ''
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return '<c r="{}"{}><v>{}</v></c>'.format(ref, style, value)
    text = escape('' if value is None else str(value))
    return '<c r="{}"{} t="inlineStr"><is><t>{}</t></is></c>'.format(ref, style, text)


def sheet_xml(headers, rows, preview_col, style_ids):
    xml_rows = []
    header_cells = [xml_cell(1, col_idx, value) for col_idx, value in enumerate(headers)]
    xml_rows.append('<row r="1">{}</row>'.format(''.join(header_cells)))
    for row_number, row in enumerate(rows, start=2):
        cells = []
        for col_idx, value in enumerate(row):
            style_id = 0
            if col_idx == preview_col:
                color = str(row[preview_col - 1]).strip().lstrip('#').upper()
                style_id = style_ids.get(color, 0)
            cells.append(xml_cell(row_number, col_idx, value, style_id=style_id))
        xml_rows.append('<row r="{}">{}</row>'.format(row_number, ''.join(cells)))

    widths = ''.join(
        '<col min="{0}" max="{0}" width="{1}" customWidth="1"/>'.format(
            idx + 1, 18 if idx == preview_col else 22)
        for idx in range(len(headers))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<cols>{}</cols>'
        '<sheetData>{}</sheetData>'
        '</worksheet>'
    ).format(widths, ''.join(xml_rows))


def styles_xml(colors):
    fills = [
        '<fill><patternFill patternType="none"/></fill>',
        '<fill><patternFill patternType="gray125"/></fill>',
    ]
    for color in colors:
        fills.append(
            '<fill><patternFill patternType="solid">'
            '<fgColor rgb="FF{0}"/><bgColor indexed="64"/>'
            '</patternFill></fill>'.format(color)
        )
    cell_xfs = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>']
    for idx in range(len(colors)):
        cell_xfs.append(
            '<xf numFmtId="0" fontId="0" fillId="{}" borderId="0" xfId="0" applyFill="1"/>'.format(idx + 2)
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>'
        '<fills count="{fills_count}">{fills}</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="{xfs_count}">{cell_xfs}</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    ).format(
        fills_count=len(fills),
        fills=''.join(fills),
        xfs_count=len(cell_xfs),
        cell_xfs=''.join(cell_xfs),
    )


def write_xlsx(path, found_headers, found_rows, legend_headers, legend_rows):
    colors = sorted({
        str(row[-2]).strip().lstrip('#').upper()
        for row in found_rows + legend_rows
        if len(str(row[-2]).strip().lstrip('#')) == 6
    })
    style_ids = {color: idx + 1 for idx, color in enumerate(colors)}

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml',
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="xml" ContentType="application/xml"/>'
                    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    '</Types>')
        zf.writestr('_rels/.rels',
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                    '</Relationships>')
        zf.writestr('xl/workbook.xml',
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    '<sheets>'
                    '<sheet name="found_classes" sheetId="1" r:id="rId1"/>'
                    '<sheet name="all_classes_palette" sheetId="2" r:id="rId2"/>'
                    '</sheets>'
                    '</workbook>')
        zf.writestr('xl/_rels/workbook.xml.rels',
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
                    '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                    '</Relationships>')
        zf.writestr('xl/styles.xml', styles_xml(colors))
        zf.writestr('xl/worksheets/sheet1.xml',
                    sheet_xml(found_headers, found_rows, preview_col=len(found_headers) - 1, style_ids=style_ids))
        zf.writestr('xl/worksheets/sheet2.xml',
                    sheet_xml(legend_headers, legend_rows, preview_col=len(legend_headers) - 1, style_ids=style_ids))


def write_excel_report(path, labels, names, confidence, kept_idx, palette):
    kept_labels = labels[kept_idx]
    unique, counts = np.unique(kept_labels, return_counts=True)

    found_headers = [
        'class_index',
        'class_name',
        'count',
        'mean_confidence',
        'color_rgb',
        'color_hex',
        'color_preview',
    ]

    rows = []
    for label, count in zip(unique.tolist(), counts.tolist()):
        mask = kept_labels == label
        conf_mean = float(confidence[kept_idx][mask].mean()) if count > 0 else 0.0
        rgb = palette[int(label)]
        rows.append([
            int(label),
            names.get(int(label), 'class_{}'.format(int(label))),
            int(count),
            conf_mean,
            color_rgb_text(rgb),
            color_hex(rgb),
            '',
        ])
    rows.sort(key=lambda row: row[2], reverse=True)

    legend_headers = [
        'class_index',
        'class_name',
        'color_rgb',
        'color_hex',
        'color_preview',
    ]
    legend_rows = []
    for idx in range(palette.shape[0]):
        rgb = palette[idx]
        legend_rows.append([
            int(idx),
            names.get(idx, 'class_{}'.format(idx)),
            color_rgb_text(rgb),
            color_hex(rgb),
            '',
        ])

    write_xlsx(path, found_headers, rows, legend_headers, legend_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    label_names = read_label_map(args.label_map)
    classes = args.classes or len(label_names)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    ckpt_classes = infer_classes_from_checkpoint(state_dict)
    if ckpt_classes:
        classes = ckpt_classes

    print('[PM INFER] reading scene:', args.scene_ply, flush=True)
    xyz, rgb = read_scene_ply(args.scene_ply)
    source_indices = np.arange(xyz.shape[0], dtype=np.int64)
    if args.input_voxel_size > 0:
        before = xyz.shape[0]
        xyz, rgb, source_indices = downsample_input(xyz, rgb, args.input_voxel_size)
        print('[PM INFER] input voxel downsample {:.4f}: {:,} -> {:,} points'.format(
            args.input_voxel_size, before, xyz.shape[0]), flush=True)
    feat = rgb.astype(np.float32) / 127.5 - 1.0

    print('[PM INFER] points:', xyz.shape[0], 'classes:', classes, flush=True)
    model_args = build_model_args(args, classes)
    model = load_pointmixer(args.checkpoint, model_args)

    pred, conf = predict_scene(
        model, xyz, feat, classes,
        args.max_points, args.min_points,
        args.block_size, args.votes,
    )

    palette = make_palette(classes)
    colorized = palette[np.clip(pred, 0, classes - 1)]
    keep = (pred != args.ignore_label) & (conf >= args.confidence_threshold)
    kept_idx = np.where(keep)[0].astype(np.int64)

    scene_name = Path(args.scene_ply).stem
    ply_path = output_dir / '{}_pointmixer_pred_colored.ply'.format(scene_name)
    labels_npy = output_dir / '{}_pointmixer_pred_labels.npy'.format(scene_name)
    labels_txt = output_dir / '{}_pointmixer_pred_labels.txt'.format(scene_name)
    source_indices_npy = output_dir / '{}_pointmixer_source_indices.npy'.format(scene_name)
    summary_csv = output_dir / '{}_pointmixer_class_summary.csv'.format(scene_name)
    summary_xlsx = output_dir / '{}_pointmixer_class_summary.xlsx'.format(scene_name)
    colors_tsv = output_dir / '{}_pointmixer_class_colors.tsv'.format(scene_name)

    np.save(labels_npy, pred)
    np.save(source_indices_npy, source_indices)
    np.savetxt(labels_txt, pred.reshape(-1, 1), fmt='%d')
    write_prediction_ply(
        ply_path,
        xyz[kept_idx],
        colorized[kept_idx],
        pred[kept_idx],
        conf[kept_idx],
        source_indices[kept_idx],
    )
    write_summary_csv(summary_csv, pred, label_names, conf, kept_idx)
    write_excel_report(summary_xlsx, pred, label_names, conf, kept_idx, palette)

    with open(colors_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['class_index', 'class_name', 'red', 'green', 'blue'])
        for idx in range(classes):
            writer.writerow([
                idx, label_names.get(idx, 'class_{}'.format(idx)),
                int(palette[idx, 0]), int(palette[idx, 1]), int(palette[idx, 2]),
            ])

    if args.write_point_csv:
        point_csv = output_dir / '{}_pointmixer_pred_points.csv'.format(scene_name)
        write_point_csv(point_csv, xyz, pred, label_names, conf, colorized, kept_idx, source_indices)
        print('[PM INFER] wrote:', point_csv, flush=True)

    print('[PM INFER] kept points: {}/{}'.format(kept_idx.shape[0], xyz.shape[0]), flush=True)
    print('[PM INFER] wrote:', ply_path, flush=True)
    print('[PM INFER] wrote:', labels_npy, flush=True)
    print('[PM INFER] wrote:', source_indices_npy, flush=True)
    print('[PM INFER] wrote:', labels_txt, flush=True)
    print('[PM INFER] wrote:', summary_csv, flush=True)
    print('[PM INFER] wrote:', summary_xlsx, flush=True)
    print('[PM INFER] done', flush=True)


if __name__ == '__main__':
    main()
