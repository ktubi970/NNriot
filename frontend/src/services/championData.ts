import { Champion, Role } from '../types';

export const DDRAGON_VERSION = '14.24.1';
export const DDRAGON_CDN = `https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VERSION}`;

export const DEFAULT_CHAMPIONS: Champion[] = [
  { id: 'Aatrox', key: '266', name: 'Aatrox', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Aatrox.png` },
  { id: 'Ahri', key: '103', name: 'Ahri', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Ahri.png` },
  { id: 'Akali', key: '84', name: 'Akali', roles: ['MID', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Akali.png` },
  { id: 'Ambessa', key: '799', name: 'Ambessa', roles: ['TOP', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Ambessa.png` },
  { id: 'Aphelios', key: '523', name: 'Aphelios', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Aphelios.png` },
  { id: 'Ashe', key: '22', name: 'Ashe', roles: ['ADC', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Ashe.png` },
  { id: 'Azir', key: '268', name: 'Azir', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Azir.png` },
  { id: 'Blitzcrank', key: '53', name: 'Blitzcrank', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Blitzcrank.png` },
  { id: 'Caitlyn', key: '51', name: 'Caitlyn', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Caitlyn.png` },
  { id: 'Camille', key: '164', name: 'Camille', roles: ['TOP', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Camille.png` },
  { id: 'Darius', key: '122', name: 'Darius', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Darius.png` },
  { id: 'Draven', key: '119', name: 'Draven', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Draven.png` },
  { id: 'Ezreal', key: '81', name: 'Ezreal', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Ezreal.png` },
  { id: 'FakerLeBlanc', key: '7', name: 'LeBlanc', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Leblanc.png` },
  { id: 'Hwei', key: '910', name: 'Hwei', roles: ['MID', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Hwei.png` },
  { id: 'Janna', key: '40', name: 'Janna', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Janna.png` },
  { id: 'Jax', key: '24', name: 'Jax', roles: ['TOP', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Jax.png` },
  { id: 'Jhin', key: '202', name: 'Jhin', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Jhin.png` },
  { id: 'Jinx', key: '222', name: 'Jinx', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Jinx.png` },
  { id: 'KaiSa', key: '145', name: "Kai'Sa", roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Kaisa.png` },
  { id: 'LeeSin', key: '64', name: 'Lee Sin', roles: ['JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/LeeSin.png` },
  { id: 'Leona', key: '89', name: 'Leona', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Leona.png` },
  { id: 'Lulu', key: '117', name: 'Lulu', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Lulu.png` },
  { id: 'Lux', key: '99', name: 'Lux', roles: ['SUPPORT', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Lux.png` },
  { id: 'Nautilus', key: '111', name: 'Nautilus', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Nautilus.png` },
  { id: 'Orianna', key: '61', name: 'Orianna', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Orianna.png` },
  { id: 'Renekton', key: '58', name: 'Renekton', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Renekton.png` },
  { id: 'Riven', key: '92', name: 'Riven', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Riven.png` },
  { id: 'Sejuani', key: '113', name: 'Sejuani', roles: ['JUNGLE', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Sejuani.png` },
  { id: 'Smolder', key: '901', name: 'Smolder', roles: ['ADC', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Smolder.png` },
  { id: 'Sylas', key: '517', name: 'Sylas', roles: ['MID', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Sylas.png` },
  { id: 'Thresh', key: '412', name: 'Thresh', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Thresh.png` },
  { id: 'Viego', key: '234', name: 'Viego', roles: ['JUNGLE', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Viego.png` },
  { id: 'Viktor', key: '112', name: 'Viktor', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Viktor.png` },
  { id: 'Vayne', key: '67', name: 'Vayne', roles: ['ADC', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Vayne.png` },
  { id: 'XinZhao', key: '5', name: 'Xin Zhao', roles: ['JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/XinZhao.png` },
  { id: 'Yasuo', key: '157', name: 'Yasuo', roles: ['MID', 'TOP', 'ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Yasuo.png` },
  { id: 'Yone', key: '777', name: 'Yone', roles: ['MID', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Yone.png` },
  { id: 'Zed', key: '238', name: 'Zed', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Zed.png` }
];

export async function fetchAllChampions(): Promise<Champion[]> {
  try {
    const res = await fetch(`https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VERSION}/data/en_US/champion.json`);
    if (!res.ok) throw new Error('Failed to fetch from DataDragon');
    const data = await res.json();
    const champions: Champion[] = Object.values(data.data).map((c: any) => {
      const tagToRoleMap: Record<string, Role[]> = {
        Fighter: ['TOP', 'JUNGLE'],
        Tank: ['TOP', 'SUPPORT', 'JUNGLE'],
        Mage: ['MID', 'SUPPORT'],
        Assassin: ['MID', 'JUNGLE'],
        Marksman: ['ADC'],
        Support: ['SUPPORT'],
      };

      const roles = new Set<Role>();
      c.tags?.forEach((tag: string) => {
        (tagToRoleMap[tag] || []).forEach(r => roles.add(r));
      });

      return {
        id: c.id,
        key: c.key,
        name: c.name,
        roles: roles.size > 0 ? Array.from(roles) : ['MID'],
        iconUrl: `${DDRAGON_CDN}/img/champion/${c.image.full}`,
        tags: c.tags,
      };
    });
    return champions.sort((a, b) => a.name.localeCompare(b.name));
  } catch (err) {
    console.warn('Using cached champions list:', err);
    return DEFAULT_CHAMPIONS;
  }
}
