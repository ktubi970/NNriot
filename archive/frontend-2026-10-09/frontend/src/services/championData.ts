import { Champion, Role } from '../types';

export const DDRAGON_VERSION = '14.24.1';
export const DDRAGON_CDN = `https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VERSION}`;

export const DEFAULT_CHAMPIONS: Champion[] = [
  { id: 'Aatrox', key: '266', name: 'Aatrox', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Aatrox.png`, winRate: 50.4, pickRate: 8.2 },
  { id: 'Ahri', key: '103', name: 'Ahri', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Ahri.png`, winRate: 51.2, pickRate: 11.5 },
  { id: 'Akali', key: '84', name: 'Akali', roles: ['MID', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Akali.png`, winRate: 49.1, pickRate: 7.8 },
  { id: 'Ambessa', key: '799', name: 'Ambessa', roles: ['TOP', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Ambessa.png`, winRate: 50.8, pickRate: 9.1 },
  { id: 'Aphelios', key: '523', name: 'Aphelios', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Aphelios.png`, winRate: 48.6, pickRate: 6.4 },
  { id: 'Ashe', key: '22', name: 'Ashe', roles: ['ADC', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Ashe.png`, winRate: 51.0, pickRate: 12.3 },
  { id: 'Azir', key: '268', name: 'Azir', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Azir.png`, winRate: 47.9, pickRate: 4.5 },
  { id: 'Blitzcrank', key: '53', name: 'Blitzcrank', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Blitzcrank.png`, winRate: 51.8, pickRate: 9.7 },
  { id: 'Caitlyn', key: '51', name: 'Caitlyn', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Caitlyn.png`, winRate: 50.2, pickRate: 15.1 },
  { id: 'Camille', key: '164', name: 'Camille', roles: ['TOP', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Camille.png`, winRate: 51.5, pickRate: 6.2 },
  { id: 'Darius', key: '122', name: 'Darius', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Darius.png`, winRate: 50.9, pickRate: 7.5 },
  { id: 'Draven', key: '119', name: 'Draven', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Draven.png`, winRate: 50.5, pickRate: 5.8 },
  { id: 'Ezreal', key: '81', name: 'Ezreal', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Ezreal.png`, winRate: 49.8, pickRate: 18.2 },
  { id: 'FakerLeBlanc', key: '7', name: 'LeBlanc', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Leblanc.png`, winRate: 49.3, pickRate: 6.9 },
  { id: 'Hwei', key: '910', name: 'Hwei', roles: ['MID', 'SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Hwei.png`, winRate: 50.1, pickRate: 7.2 },
  { id: 'Janna', key: '40', name: 'Janna', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Janna.png`, winRate: 52.4, pickRate: 7.1 },
  { id: 'Jax', key: '24', name: 'Jax', roles: ['TOP', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Jax.png`, winRate: 50.7, pickRate: 8.9 },
  { id: 'Jhin', key: '202', name: 'Jhin', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Jhin.png`, winRate: 51.7, pickRate: 16.4 },
  { id: 'Jinx', key: '222', name: 'Jinx', roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Jinx.png`, winRate: 51.1, pickRate: 14.8 },
  { id: 'KaiSa', key: '145', name: "Kai'Sa", roles: ['ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Kaisa.png`, winRate: 49.6, pickRate: 19.5 },
  { id: 'LeeSin', key: '64', name: 'Lee Sin', roles: ['JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/LeeSin.png`, winRate: 49.2, pickRate: 16.7 },
  { id: 'Leona', key: '89', name: 'Leona', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Leona.png`, winRate: 51.3, pickRate: 11.2 },
  { id: 'Lulu', key: '117', name: 'Lulu', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Lulu.png`, winRate: 50.9, pickRate: 9.3 },
  { id: 'Lux', key: '99', name: 'Lux', roles: ['SUPPORT', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Lux.png`, winRate: 50.6, pickRate: 12.8 },
  { id: 'Nautilus', key: '111', name: 'Nautilus', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Nautilus.png`, winRate: 50.1, pickRate: 10.4 },
  { id: 'Orianna', key: '61', name: 'Orianna', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Orianna.png`, winRate: 49.5, pickRate: 8.1 },
  { id: 'Renekton', key: '58', name: 'Renekton', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Renekton.png`, winRate: 49.7, pickRate: 9.2 },
  { id: 'Riven', key: '92', name: 'Riven', roles: ['TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Riven.png`, winRate: 51.4, pickRate: 5.4 },
  { id: 'Sejuani', key: '113', name: 'Sejuani', roles: ['JUNGLE', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Sejuani.png`, winRate: 50.8, pickRate: 4.8 },
  { id: 'Smolder', key: '901', name: 'Smolder', roles: ['ADC', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Smolder.png`, winRate: 48.9, pickRate: 7.3 },
  { id: 'Sylas', key: '517', name: 'Sylas', roles: ['MID', 'JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/Sylas.png`, winRate: 50.3, pickRate: 11.4 },
  { id: 'Thresh', key: '412', name: 'Thresh', roles: ['SUPPORT'], iconUrl: `${DDRAGON_CDN}/img/champion/Thresh.png`, winRate: 51.2, pickRate: 13.9 },
  { id: 'Viego', key: '234', name: 'Viego', roles: ['JUNGLE', 'MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Viego.png`, winRate: 50.5, pickRate: 12.1 },
  { id: 'Viktor', key: '112', name: 'Viktor', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Viktor.png`, winRate: 50.7, pickRate: 8.4 },
  { id: 'Vayne', key: '67', name: 'Vayne', roles: ['ADC', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Vayne.png`, winRate: 51.0, pickRate: 8.6 },
  { id: 'XinZhao', key: '5', name: 'Xin Zhao', roles: ['JUNGLE'], iconUrl: `${DDRAGON_CDN}/img/champion/XinZhao.png`, winRate: 51.6, pickRate: 6.9 },
  { id: 'Yasuo', key: '157', name: 'Yasuo', roles: ['MID', 'TOP', 'ADC'], iconUrl: `${DDRAGON_CDN}/img/champion/Yasuo.png`, winRate: 49.9, pickRate: 13.2 },
  { id: 'Yone', key: '777', name: 'Yone', roles: ['MID', 'TOP'], iconUrl: `${DDRAGON_CDN}/img/champion/Yone.png`, winRate: 49.4, pickRate: 12.7 },
  { id: 'Zed', key: '238', name: 'Zed', roles: ['MID'], iconUrl: `${DDRAGON_CDN}/img/champion/Zed.png`, winRate: 50.0, pickRate: 9.8 }
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
        winRate: 50.0 + (Math.sin(parseInt(c.key, 10)) * 2.5),
        pickRate: 5.0 + (Math.cos(parseInt(c.key, 10)) * 4.0),
      };
    });
    return champions.sort((a, b) => a.name.localeCompare(b.name));
  } catch (err) {
    console.warn('Using cached champions list:', err);
    return DEFAULT_CHAMPIONS;
  }
}
