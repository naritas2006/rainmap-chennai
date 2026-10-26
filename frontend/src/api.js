const get = async (path) => {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
};

export const api = {
  neighbourhoods: () => get("/neighbourhoods"),
  summary: (n) => get(`/neighbourhoods/${n}`),
  buildings: (n) => get(`/neighbourhoods/${n}/buildings`),
  routes: (n) => get(`/neighbourhoods/${n}/routes`),
  clusters: (n) => get(`/neighbourhoods/${n}/clusters`),
  rainfall: () => get("/rainfall"),
  ml: () => get("/ml"),
};
