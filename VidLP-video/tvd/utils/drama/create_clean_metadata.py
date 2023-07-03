import os
import sys
import json
import pandas as pd
import random
random.seed(5)


root_dir = "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/videos/subset_0/annotations-biaozhu25001"
out_folder = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/'

train_split_path = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/train.csv'
val_split_path = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/val.csv'

def get_vids(p):
    vids = set(pd.read_csv(p)['Name'])
    vids_prefix = [vid.split('/')[-1].split('.')[0] for vid in vids]
    return vids_prefix
    # vids_list = []
    # for vid in vids:
    #     if vid in vids_list:
    #         pass
    #     else:
    #         vids_list.append(vid)
    # return vids_list

train_vids = get_vids(train_split_path)
val_vids = get_vids(val_split_path)
val_vids = list(set(val_vids) - set(train_vids))
train_val_vids = set(val_vids) | set(train_vids)
print('pretraining corpus: train {} val {} '.format(len(train_vids), len(val_vids)))
N_test=45
val_vids = sorted(val_vids)
random.shuffle(val_vids)
test_vids = val_vids[-N_test:]
val_vids = val_vids[:-N_test]
all_vids = {'train': train_vids, 'val': val_vids, 'test': test_vids}
print('train_ids', len(train_vids), 'val_ids', len(val_vids), 'test_vids', len(test_vids))
train_val_vids_for_annos = []
for split_name, split_vids in all_vids.items():
    seg_id_list=[]
    vid_p_list = []
    for file in os.listdir(root_dir):
        assert file.endswith('.json')
        file_path = os.path.join(root_dir, file)
        d = json.load(open(file_path))
        num = len(d['timestamps'])
        vid = file.split('.')[0]
        train_val_vids_for_annos.append(vid)
        if split_name=='train' and (vid in val_vids or vid in test_vids):
            # print('vid {} not exists'.format(vid))
            continue
        if split_name == 'val' and vid not in val_vids:
            continue
        if split_name == 'test' and vid not in test_vids:
            continue
        vid_p = os.path.join('subset_0/data/', vid + '.mp4')
        seg_id_list.extend(list(range(num)))
        vid_p_list.extend([vid_p] * num)
    print('split {}, {} videos, {} annos'.format(split_name, len(set(vid_p_list)), len(vid_p_list)))
    data_dict = {'Name': vid_p_list, 'segment_id': seg_id_list, 'type': ['annotations-biaozhu25001']*len(vid_p_list)}
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(os.path.join(out_folder, split_name+'.csv'), index=False)

print('missing vids:')
print(set(train_val_vids_for_annos)-set(train_val_vids))


# anno_v1 = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/videos/subset_0/annotations'
# anno_v2 = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/videos/subset_0/annotations-biaozhu25001'
# print('anno_v1-anno_v2')
# print(set(os.listdir(anno_v1)) - set(os.listdir(anno_v2)))
# {'v0029vzl555.json', 'a0015mp5ogo.json', 'w0034hw1rzu.json', 'e00276m9qyi.json', 'a0029au9icx.json', 'x00152v3g8l.json', 'o00420xgcmn.json', 'i00116gky2k.json', 'p00428r65kh.json', 'p0042fidl4o.json', 'e0042i2xkjk.json', 'k0011oql8xq.json', 'f0022wu5czg.json', 'r0041yuriub.json', 'a0043pnim4g.json', 'd0042hysg2p.json', 'i0029maiazd.json', 'u0027uzhl76.json', 'f0042wno9r8.json', 'j0029cy8wk5.json', 'l00424dguuo.json', 'f0042ukjsyr.json', 'x00157qqx5f.json', 'e0041ylndex.json', 'n00340dun11.json', 'n0034bwcztw.json', 'f0029hdinzt.json', 'a0027ioq0ju.json', 'b004126s6c0.json', 'h0042h6woer.json', 'b0029nz31ek.json', 'b0043zy0ptp.json', 'i0043eyd9fb.json', 'y0022v8dmex.json', 'v0029owunvd.json', 'j0029nx3ulk.json', 'w0041y5qkyp.json', 'b00288n6okn.json', 'e0041qpvva2.json', 'g0015l6nl7y.json', 'p00295vk3ui.json', 'j0043c2hgxf.json', 'n0011kwlmhe.json', 'w0029xb859g.json', 'z0042r0t6cd.json', 'n0014ylzeed.json', 'b00144v9t8e.json', 's0034er8neh.json', 'i0027yv0azw.json', 'c0022vtws9c.json', 'j00421otcyh.json', 'i0034bcdim8.json', 'h0042zpt820.json', 'i0043p38m1e.json', 'p0014nrk8ej.json', 'x0027jdlfaj.json', 'm00413w2cya.json', 'h0029h8687d.json', 'q0041y9baos.json', 'c0041rg972a.json', 'm00410lavws.json', 'h0029iinpds.json', 'o0042nih0pe.json', 'i004361c5b5.json', 'r0027kjzsii.json', 'u004194h3co.json', 'a0022liude8.json', 'b0015zo9tb6.json', 't0034w9702m.json', 'l0042nfexkt.json', 'z0014m4v04s.json', 'r0042hob53c.json', 's00436s2lgc.json', 'a0034pg5dtz.json', 'v0028fp8hg9.json', 'w0041dmwdmv.json', 'd0022zybznx.json', 'h0029zvizkz.json', 'e0022gmexh4.json', 'x00415kcyeq.json', 's0041vcj8xp.json', 'l0022xjkclk.json', 'k0011m3pedl.json', 'h0029vb2uwg.json', 'c0042ul93yr.json', 't0041duk499.json', 'm00417bswwa.json', 'w0041oweo2e.json', 'a0011r4ko4f.json', 'd0034r7nyg6.json', 'p0027r5meam.json', 'd0042v9lp57.json', 'b00423him94.json', 'x0027qpvh65.json', 'm0028gojmo8.json', 'x002986dezc.json', 'f0042s2x3sf.json', 'j00280s4i6a.json', 'a0041d6ry2m.json', 'x004254svwy.json', 'e0041llhf86.json', 'e0043s91bxj.json', 't0034tizt37.json', 'p004142ys98.json', 'h0029v8ivgr.json', 'n0022imbu63.json', 'x0022i56e6y.json', 'i0034awhjar.json', 'x0042bm22ck.json', 'x0011iehk9n.json', 'p0042v87ggw.json', 'a0042cip6z7.json', 'a0027tvea57.json', 'x00424lbviw.json', 'f0015du7k9c.json', 'k00415t1ufc.json', 'n00112tuap1.json', 'm0042pxnhdx.json', 's00223zegmb.json', 'm0029ufc5c5.json', 'y0042wljsxb.json', 'p00280sml2t.json', 'd004109qjtc.json', 'a00420ngwe1.json', 'e00222zun3z.json', 'i0041l6bsnx.json', 'w0029lt4ea5.json', 'd0042o7aep5.json', 's0042bhcb8c.json', 'p00436irxau.json', 'e00342qmphc.json', 'i0029wy1zzm.json', 'n001519f0p7.json', 'v0042iqu605.json', 'd0041k8opcb.json', 't003005xrj5.json', 'x0042tmn6zo.json', 'e00427it43e.json', 'h0034panihg.json', 'j0029max1bd.json', 'c004379ne2n.json', 'p0041s8wiwo.json', 'b0029shkvvr.json', 'b00424vuet6.json', 's0028kfnu4c.json', 'm0029md0eon.json', 'v00134bnlyg.json', 'z0042gpkbwn.json', 'e0034u1ipkc.json', 'x0014xjv93e.json', 'x002968ousy.json', 'z0042vyu3h0.json', 'n0027uqt9qi.json', 'r0041br5gam.json', 'p00427gy121.json', 'y0042bfu0rf.json', 'd0022anhnnn.json', 'n004221bsu4.json', 'r00290xymzd.json', 'z0011gwzbqi.json', 'l0034jlgyrt.json', 'i0034l6mr1b.json', 's0034x9y2xu.json', 'v0041yy6m2o.json', 'g00222lb5w2.json', 'f0034ox8mfs.json', 'm0042ex7iuy.json'}
# print('anno_v2-anno_v1')
# print(set(os.listdir(anno_v2)) - set(os.listdir(anno_v1)))
# {'y00255l8ta2.json', 'g0037gik6fn.json', 'l0025aurklv.json', 't00426tl2hm.json', 'd0015az4mmz.json', 'd0029yup6iq.json', 't0042saidwt.json', 'i0042b44tys.json', 'j0025qoql4d.json', 'w0037qc2z8v.json', 'x0020rlxsvx.json', 'k00213syhpm.json', 'v00370k13c4.json', 'l00290ouv3a.json', 'i00215x5fdg.json', 't0028dpjqa2.json', 'j001512qgc5.json', 'u0037n6pqaw.json', 'h0037h1lyxy.json', 'v0033owt1nq.json', 'g0040g8l1do.json', 'i0021kgu973.json', 'e00216q5zij.json', 'c0021xkaqyr.json', 'f00421bcsu3.json', 'p0021a6gq1f.json', 's00210sgvzx.json', 'u00274tc98c.json', 'e001505rewn.json', 'l00372wctbn.json', 'a0025mqd5pr.json', 'v0037o4vezs.json', 'r0029jn2uag.json', 'j0025ibdhn0.json', 'x0021cx9qlh.json', 'x0027x05pb4.json', 'r0033394bny.json', 'k0037f9vky9.json', 'h0020coqffy.json', 'm0021ymy90k.json', 'y0021mci3za.json', 'l0021pvqkmr.json', 'y002587132q.json', 'x0024etsjzj.json', 'h0042h8f0m3.json', 'u0029pmcyw8.json', 's0037n2hmk0.json', 'q0021nsqke8.json', 't00211rd4ht.json', 'q0015hnvrw5.json', 'm00156fvboj.json', 'r0042ydu6c4.json', 'y0015v4lz23.json', 'u0029ajkvb7.json', 'q00215ukkfj.json', 't00218qbzx8.json', 'g003318p1u7.json', 'u0021aj1xhw.json', 'b0042w5v16c.json', 'v00256fw5lf.json', 'k0027alzglz.json', 'r0021lvb565.json', 'o0029m3tyqn.json', 'l0025e0ofij.json', 'v0021ith5je.json', 'r00241k92a8.json', 'q0033ik62nu.json', 'n0021sggkel.json', 'q0028lxlx09.json', 'r0037j31c9i.json', 'n0037fdltlh.json', 'q00379k3jh8.json', 'h00405iak7x.json', 'c0015a31fa7.json', 'w00150g1h9a.json', 'l0042o0pnoc.json', 'q0025izgvl9.json', 'j0024wb93fh.json', 'w002551m9n6.json', 'n0014td4o4b.json', 'a0025vdt5q4.json', 'x0021wn5z6n.json', 'c0037rmxbeu.json', 'a0037u2m8k3.json', 'k00401q4h0l.json', 'f00210uevf9.json', 'p0029wpp8z1.json', 'v0015nj64sw.json', 'd0021do8f2w.json', 'o0042cdsaio.json', 'n0020u6kg2o.json', 'z00400es1fq.json', 's0043kqys2y.json', 'v0043elc9hw.json', 'z0029u84cwe.json', 'n0024lagymu.json', 's0028ngobey.json', 'r0042v9in9w.json', 'n0024lgic61.json', 'w0016t4sday.json', 'h0021r4tl67.json', 'h0020mwxga5.json', 'h003735rgx7.json', 'd0021156r7f.json', 'h0021c78xep.json', 'y0021qln2ek.json', 'z0028fl8y25.json', 'r002403p3rn.json', 'e0037d6d9ev.json', 'z0037vyi8k2.json', 'w0029pkei6o.json', 'r0021yblcvg.json', 'b0037ajvuiq.json', 'w0037geymhz.json', 'm0037oabews.json', 'm0042f61xx3.json', 't0042ak4e7v.json', 'c0028yxx4kb.json', 'l0021w1vae8.json', 'u0037b2qa38.json', 'l00217sv07w.json', 'k0042icazfo.json', 'm0021vumafk.json', 'r0040ya4c2g.json', 'b0021qqdmy4.json', 'v0033xvx01z.json', 'f004094g7fc.json', 'l00212kw7sb.json', 'b0037ubx72p.json', 'u00423nbf3w.json', 'q0021a8pg4e.json', 's00372kscox.json', 'j002183mc49.json', 'b0021evst1n.json', 'a0025uezlv6.json', 'o0033qrbl7q.json', 'f0021tzwzcb.json', 'n0037sqp335.json', 'f0015j6lofk.json', 'v00335imgpz.json', 't0037ktjciy.json', 'l00211d0eqi.json', 'm0021nev7og.json', 'y0021i6fzsr.json', 'g0025tadva0.json', 'r0028fgh8m4.json', 'z0037w034ta.json', 'p0025i31q4t.json', 'n004024fn32.json', 'r0021an800n.json', 'f0025l0r635.json', 'h0033km1nh2.json', 'v0042gps5g6.json'}